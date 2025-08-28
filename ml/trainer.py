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
    if user_settings.get('data_dirs'):
        data_dirs = user_settings['data_dirs']
        
        if is_inference_dataloader:
            # InferenceTraceSeqEWDataloader expects data_dirs as a list
            if isinstance(data_dirs, dict):
                data_dirs_list = list(data_dirs.values())
            else:
                data_dirs_list = data_dirs
            config['data']['init_args']['data_dirs'] = data_dirs_list
        else:
            # TraceSeqEWDataloader expects data_dirs as a dictionary
            if isinstance(data_dirs, list):
                data_dirs_dict = {}
                for path in data_dirs:
                    key = Path(path).name
                    data_dirs_dict[key] = path
            else:
                data_dirs_dict = data_dirs
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
    else:
        # TraceSeqEWDataloader specific parameters
        if user_settings.get('label_dir'):
            config['data']['init_args']['label_dir'] = user_settings['label_dir']
    
    # Handle scaler path from checkpoint directory
    if user_settings.get('ckpt_path'):
        ckpt_path_str = user_settings['ckpt_path']
        ckpt_path = Path(ckpt_path_str)
        version_dir = ckpt_path.parent.parent
        
        # Set checkpoint path for the trainer to load model weights
        if 'ckpt_path' not in config:
            config['ckpt_path'] = ckpt_path_str
            print(f"Using checkpoint: {ckpt_path_str}")
        
        # Find and set scaler path for dataloader
        scaler_files = list(version_dir.glob("*.pth"))
        if scaler_files:
            scaler_path = str(scaler_files[0])
            config['data']['init_args']['scaler_path'] = scaler_path
            print(f"Using scaler: {scaler_path}")
        else:
            print(f"Warning: No .pth scaler files found in {version_dir}")
    
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
    import os
    
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
                
                # Find the main config file in sys.argv
                main_config_path = None
                for i, arg in enumerate(sys.argv):
                    if arg == "--config" and i + 1 < len(sys.argv):
                        potential_config = sys.argv[i + 1]
                        if subcommand in potential_config and "training" in potential_config:
                            main_config_path = potential_config
                            main_config_index = i + 1
                            break
                
                if main_config_path:
                    # Create temporary config with user settings injected
                    temp_config_path = create_temp_config_with_user_settings(
                        main_config_path, user_settings, subcommand
                    )
                    
                    # Replace the main config path in sys.argv
                    sys.argv[main_config_index] = temp_config_path
            
            return user_settings
        except Exception as e:
            print(f"Error loading user config {user_config_path}: {e}")
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