import os
import sys
import yaml
import torch
import warnings
from pathlib import Path
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities import disable_possible_user_warnings

# Suppress UserWarnings from PyTorch
warnings.filterwarnings("ignore", category=UserWarning)


def create_temp_config_with_user_settings(base_config_path, user_settings, subcommand):
    """Create a temporary config file with user settings injected."""
    import tempfile
    
    # Load the base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check which dataloader is being used to handle parameters correctly
    dataloader_class = config['data']['class_path']
    is_inference_dataloader = 'InferenceTraceSeqEWDataloader' in dataloader_class
    
    # Inject user settings into the config  
    data_dirs = user_settings.get('data_dirs')
    if data_dirs:  # Handle truthy values (not None, not empty list/dict)
        if is_inference_dataloader:
            # InferenceTraceSeqEWDataloader expects data_dirs as a list
            if isinstance(data_dirs, dict):
                data_dirs_list = list(data_dirs.values())
            elif isinstance(data_dirs, list):
                data_dirs_list = data_dirs
            else:
                # Convert single path to list
                data_dirs_list = [str(data_dirs)]
            config['data']['init_args']['data_dirs'] = data_dirs_list
        else:
            # TraceSeqEWDataloader expects data_dirs as a dictionary
            if isinstance(data_dirs, list):
                data_dirs_dict = {}
                for path in data_dirs:
                    if path:  # Skip None/empty paths
                        key = Path(str(path)).name
                        data_dirs_dict[key] = str(path)
            elif isinstance(data_dirs, dict):
                data_dirs_dict = data_dirs
            else:
                # Convert single path to dict
                path_str = str(data_dirs)
                data_dirs_dict = {Path(path_str).name: path_str}
            config['data']['init_args']['data_dirs'] = data_dirs_dict
    
    # Handle different parameters based on dataloader type
    if is_inference_dataloader:
        # InferenceTraceSeqEWDataloader specific parameters
        if 'drv_snp' in user_settings:
            config['data']['init_args']['drv_snp'] = user_settings['drv_snp']
        if 'odt_snp' in user_settings:
            config['data']['init_args']['odt_snp'] = user_settings['odt_snp']
        if 'bound_path' in user_settings:
            config['data']['init_args']['bound_path'] = user_settings['bound_path']
        if 'training_config_keys' in user_settings:
            config['data']['init_args']['training_config_keys'] = user_settings['training_config_keys']
    else:
        # TraceSeqEWDataloader specific parameters
        if user_settings.get('label_dir'):
            config['data']['init_args']['label_dir'] = user_settings['label_dir']
    
    # Handle logger save_dir override from user settings
    if user_settings.get('save_dir'):
        if 'trainer' not in config:
            config['trainer'] = {}
        if 'logger' not in config['trainer']:
            config['trainer']['logger'] = {}
        if 'init_args' not in config['trainer']['logger']:
            config['trainer']['logger']['init_args'] = {}
        
        config['trainer']['logger']['init_args']['save_dir'] = user_settings['save_dir']
        print(f"Using user-specified logger save_dir: {user_settings['save_dir']}")
    
    # Handle prediction writer file_prefix override from user settings
    if user_settings.get('file_prefix'):
        if 'trainer' in config and 'callbacks' in config['trainer']:
            for callback in config['trainer']['callbacks']:
                if 'EWPredictionWriter' in callback.get('class_path', ''):
                    if 'init_args' not in callback:
                        callback['init_args'] = {}
                    callback['init_args']['file_prefix'] = user_settings['file_prefix']
                    print(f"Using user-specified file_prefix: {user_settings['file_prefix']}")
                    break
    
    # Handle scaler path from checkpoint directory
    if user_settings.get('ckpt_path'):
        ckpt_path_str = user_settings['ckpt_path']
        ckpt_path = Path(ckpt_path_str)
        version_dir = ckpt_path.parent.parent
        
        # Set checkpoint path for the trainer to load model weights
        if 'ckpt_path' not in config:
            config['ckpt_path'] = ckpt_path_str
            print(f"Using checkpoint: {ckpt_path_str}")
        
        # Clear any ckpt_path from model init_args to avoid double loading
        # Lightning CLI handles checkpoint loading when ckpt_path is set at top level
        if subcommand in ["predict", "test", "validate"]:
            # Remove ckpt_path from model init_args to prevent manual loading in module
            if 'model' in config and 'init_args' in config['model']:
                model_init_args = config['model']['init_args']
                
                # Remove from top-level model init_args
                if 'ckpt_path' in model_init_args:
                    del model_init_args['ckpt_path']
                    print("Removed ckpt_path from model init_args - Lightning CLI will handle checkpoint loading")
                
                # Also check nested model structure (e.g., EyeWidthRegressor containing another model)
                if 'model' in model_init_args and isinstance(model_init_args['model'], dict) and 'init_args' in model_init_args['model']:
                    nested_init_args = model_init_args['model']['init_args']
                    if 'ckpt_path' in nested_init_args:
                        del nested_init_args['ckpt_path']
                        print("Removed ckpt_path from nested model init_args - Lightning CLI will handle checkpoint loading")
        
        # Find and set scaler path for dataloader
        scaler_files = list(version_dir.glob("*.pth"))
        if scaler_files:
            scaler_path = str(scaler_files[0])
            config['data']['init_args']['scaler_path'] = scaler_path
            print(f"Using scaler: {scaler_path}")
        else:
            print(f"Warning: No .pth scaler files found in {version_dir}")
    
    # Validate final config before creating temp file
    try:
        # Ensure data section exists and has required structure
        if 'data' not in config:
            raise ValueError("Config missing 'data' section")
        if 'init_args' not in config['data']:
            config['data']['init_args'] = {}
        
        # For inference dataloader, ensure data_dirs is not None
        if is_inference_dataloader:
            if 'data_dirs' not in config['data']['init_args'] or config['data']['init_args']['data_dirs'] is None:
                print("Warning: data_dirs not set for inference dataloader, using empty list")
                config['data']['init_args']['data_dirs'] = []
        
        print(f"Final config validation passed for {subcommand}")
    except Exception as e:
        print(f"Config validation error: {e}")
        raise
    
    # Create temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix=f'temp_{subcommand}_')
    try:
        with os.fdopen(temp_fd, 'w') as f:
            yaml.safe_dump(config, f)
        return temp_path
    except Exception as e:
        os.close(temp_fd)
        os.unlink(temp_path)
        raise e


def preprocess_user_config():
    """
    Preprocess user config and modify sys.argv to include model config if specified.
    This needs to happen before Lightning CLI initialization.
    """
    # Check if we have a user_config argument
    user_config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--user_config" and i + 1 < len(sys.argv):
            user_config_path = Path(sys.argv[i + 1])
            break
    
    # If no user_config specified, try default paths based on subcommand
    if not user_config_path:
        if len(sys.argv) > 1:
            subcommand = sys.argv[1]
            if subcommand == "fit":
                user_config_path = Path("configs/user/train_setting.yaml")
            elif subcommand == "predict":
                user_config_path = Path("configs/user/infer_setting.yaml")
            elif subcommand == "test":
                user_config_path = Path("configs/user/test_setting.yaml")
    
    # Load user settings if the file exists
    if user_config_path and user_config_path.exists():
        print(f"Loading user configuration: {user_config_path}")
        try:
            with open(user_config_path, 'r') as f:
                user_settings = yaml.safe_load(f)
            
            # Validate user_settings is not None
            if user_settings is None:
                print(f"Warning: User config {user_config_path} is empty or invalid")
                return None
            
            # For predict/test modes, find training config from ckpt_path and inject it
            if len(sys.argv) > 1 and sys.argv[1] in ["predict", "test"]:
                if user_settings.get('ckpt_path'):
                    ckpt_path = Path(user_settings['ckpt_path'])
                    version_dir = ckpt_path.parent.parent

                    # Find training config
                    training_config_path = version_dir / 'config.yaml'
                    if not training_config_path.exists():
                        possible_configs = list(version_dir.glob("*.yaml"))
                        if not possible_configs:
                            raise FileNotFoundError(f"No training config (.yaml) found in {version_dir}")
                        
                        filtered_configs = [p for p in possible_configs if p.name != 'hparams.yaml']
                        if len(filtered_configs) == 1:
                            training_config_path = filtered_configs[0]
                        elif len(filtered_configs) == 0 and len(possible_configs) == 1:
                            training_config_path = possible_configs[0]
                        else:
                            configs_to_report = filtered_configs if filtered_configs else possible_configs
                            raise FileNotFoundError(f"Ambiguous training config in {version_dir}. Found: {[p.name for p in configs_to_report]}.")
                    
                    print(f"Loading training configuration: {training_config_path}")
                    
                    # Insert training config before main config
                    config_index = -1
                    for i, arg in enumerate(sys.argv):
                        if arg == "--config":
                            config_index = i
                            break
                    
                    if config_index != -1:
                        sys.argv.insert(config_index, "--config")
                        sys.argv.insert(config_index + 1, str(training_config_path))
                    else:
                        sys.argv.extend(["--config", str(training_config_path)])
            
            # For predict/test modes, create a temporary config with user settings pre-injected
            if len(sys.argv) > 1 and sys.argv[1] in ["predict", "test"]:
                subcommand = sys.argv[1]
                
                # Find the last config file in sys.argv (inference config)
                main_config_path = None
                main_config_index = -1
                for i, arg in enumerate(sys.argv):
                    if arg == "--config" and i + 1 < len(sys.argv):
                        # Take the last config file (inference config, not training config)
                        main_config_path = sys.argv[i + 1]
                        main_config_index = i + 1
                
                if main_config_path:
                    # Extract training_config_keys from the training config
                    with open(training_config_path, 'r') as f:
                        training_config = yaml.safe_load(f)
                    
                    # Look for config_keys in the loaded training config
                    training_config_keys = None
                    if 'data' in training_config and 'init_args' in training_config['data']:
                        # This path is for TraceSeqEWDataloader during training
                        if 'config_keys' in training_config['data']['init_args']:
                            training_config_keys = training_config['data']['init_args']['config_keys']
                    
                    # Fallback for older configs where it might be at the top level
                    if not training_config_keys and 'config_keys' in training_config:
                        training_config_keys = training_config['config_keys']
                    
                    # Inject training_config_keys into user settings for the dataloader
                    if training_config_keys:
                        user_settings['training_config_keys'] = training_config_keys
                        print(f"Found and injected training_config_keys: {training_config_keys}")

                    # Create temporary config with user settings injected
                    temp_config_path = create_temp_config_with_user_settings(
                        main_config_path, user_settings, subcommand
                    )
                    
                    # Replace the main config path in sys.argv
                    sys.argv[main_config_index] = temp_config_path
            
            return user_settings
        except Exception as e:
            print(f"Error loading user config {user_config_path}: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None
    
    return None


class CustomLightningCLI(LightningCLI):
    """Custom CLI to handle checkpoint and scaler path resolution for predictions and testing."""
    def __init__(self, *args, **kwargs):
        if 'predict' in sys.argv or 'test' in sys.argv:
            kwargs['save_config_callback'] = None
        
        # Preprocess user config before Lightning CLI initialization
        self.user_settings = preprocess_user_config()
        super().__init__(*args, **kwargs)
    
    def add_arguments_to_parser(self, parser):
        # Add user config argument for all modes
        parser.add_argument("--user_config", type=str, help="Path to user configuration file")

    def before_instantiate_classes(self):
        # Pass ignore_snp flag from model to datamodule for all subcommands
        if self.subcommand and hasattr(self.config, self.subcommand):
            subcommand_config = getattr(self.config, self.subcommand)
            if hasattr(subcommand_config, 'model') and hasattr(subcommand_config.model, 'init_args') and \
               hasattr(subcommand_config.model.init_args, 'ignore_snp'):
                
                ignore_snp = subcommand_config.model.init_args.ignore_snp
                
                # Pass to nested model (EyeWidthRegressor) if it exists
                if hasattr(subcommand_config.model.init_args, 'model') and hasattr(subcommand_config.model.init_args.model, 'init_args'):
                    subcommand_config.model.init_args.model.init_args.ignore_snp = ignore_snp

                # Pass to datamodule
                if hasattr(subcommand_config, 'data') and hasattr(subcommand_config.data, 'init_args'):
                    subcommand_config.data.init_args.ignore_snp = ignore_snp

    def after_instantiate_classes(self):
        """Set output directory for EWPredictionWriter callbacks from logger directory."""
        if self.trainer and self.trainer.logger and hasattr(self.trainer.logger, 'log_dir'):
            logger_dir = self.trainer.logger.log_dir
            
            # Find EWPredictionWriter callbacks and set their output directory
            for callback in self.trainer.callbacks:
                if hasattr(callback, '__class__') and 'EWPredictionWriter' in callback.__class__.__name__:
                    if hasattr(callback, 'set_output_dir'):
                        callback.set_output_dir(logger_dir)
                        print(f"Set EWPredictionWriter output directory to: {logger_dir}")


def setup_torch_compile_fallback():
    """Fallback to eager mode if torch.compile fails."""
    try:
        @torch.compile
        def test_fn(x):
            return x + 1
        test_fn(torch.randn(2, 2))
    except Exception:
        print("Disabling torch.compile and falling back to eager mode.")
        import os
        os.environ['TORCH_COMPILE_DISABLE'] = '1'
        
def cli_main():
    setup_torch_compile_fallback()
    cli = CustomLightningCLI(datamodule_class=None)

if __name__ == "__main__":
    import logging
    logging.getLogger("tensorboardX").setLevel(logging.ERROR)

    warnings.filterwarnings('ignore')
    disable_possible_user_warnings()
    torch.set_float32_matmul_precision('medium')
    cli_main()