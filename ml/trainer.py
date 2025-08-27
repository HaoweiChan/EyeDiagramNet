import sys
import yaml
import torch
import warnings
from pathlib import Path
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities import disable_possible_user_warnings
from lightning.pytorch.callbacks import ModelCheckpoint

# Suppress UserWarnings from PyTorch
warnings.filterwarnings("ignore", category=UserWarning)


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
                            # Raise error if ambiguity cannot be resolved
                            configs_to_report = filtered_configs if filtered_configs else possible_configs
                            raise FileNotFoundError(f"Ambiguous training config in {version_dir}. Found: {[p.name for p in configs_to_report]}. Please create a 'config.yaml' file in that directory or ensure only one non-hparams config file is present.")
                    
                    # Insert training config before main config
                    # LightningCLI merges configs, so training config should come first
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
            
            return user_settings
        except Exception as e:
            print(f"Warning: Could not load user config {user_config_path}: {e}")
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
        if self.subcommand in ["predict", "test"]:
            # Apply user settings to predict/test configuration
            if self.user_settings:
                config_attr = getattr(self.config, self.subcommand)
                
                # Apply user settings to data configuration if specified
                if self.user_settings.get('data_dirs'):
                    config_attr.data.init_args.data_dirs = self.user_settings['data_dirs']
                if self.user_settings.get('batch_size'):
                    config_attr.data.init_args.batch_size = self.user_settings['batch_size']
                if self.user_settings.get('drv_snp'):
                    config_attr.data.init_args.drv_snp = self.user_settings['drv_snp']
                if self.user_settings.get('odt_snp'):
                    config_attr.data.init_args.odt_snp = self.user_settings['odt_snp']
                if self.user_settings.get('bound_path'):
                    config_attr.data.init_args.bound_path = self.user_settings['bound_path']
                if self.user_settings.get('ignore_snp') is not None:
                    config_attr.data.init_args.ignore_snp = self.user_settings['ignore_snp']
                
                # Apply user settings to callback configuration (for predict mode)
                if self.subcommand == "predict" and hasattr(config_attr, 'trainer') and hasattr(config_attr.trainer, 'callbacks'):
                    for callback in config_attr.trainer.callbacks:
                        if hasattr(callback, 'class_path') and 'PredictionWriter' in callback.class_path:
                            if self.user_settings.get('file_prefix'):
                                callback.init_args.file_prefix = self.user_settings['file_prefix']
                                break
            
                # Handle predict and test modes - load checkpoint and scaler from user settings
                config_attr = getattr(self.config, self.subcommand)
                
                if self.user_settings.get('ckpt_path'):
                    ckpt_path_str = self.user_settings['ckpt_path']
                    # The ckpt_path for the trainer should be set from user config
                    config_attr.ckpt_path = ckpt_path_str
                    print(f"Set ckpt_path from user settings: {ckpt_path_str}")
                    
                    # Find scaler in the model version directory, derived from ckpt_path
                    version_dir = Path(ckpt_path_str).parent.parent
                    scaler_files = list(version_dir.glob("*.pth"))
                    if scaler_files:
                        scaler_path = scaler_files[0]  # Use first scaler found
                        config_attr.data.init_args.scaler_path = str(scaler_path)
                        print(f"Found and set scaler_path to: {scaler_path}")
                    else:
                        print(f"Warning: No .pth scaler files found in {version_dir}")
                else:
                    print("Warning: 'ckpt_path' not found in user settings. Checkpoint and scaler paths may not be set correctly.")

        # Pass ignore_snp flag from model to datamodule for all subcommands
        if self.subcommand and hasattr(self.config, self.subcommand):
            subcommand_config = getattr(self.config, self.subcommand)
            if hasattr(subcommand_config, 'model') and hasattr(subcommand_config.model, 'init_args') and \
               hasattr(subcommand_config.model.init_args, 'ignore_snp'):
                
                ignore_snp = subcommand_config.model.init_args.ignore_snp
                
                # Pass to nested model (EyeWidthRegressor) if it exists
                if hasattr(subcommand_config.model.init_args, 'model') and hasattr(subcommand_config.model.init_args.model, 'init_args'):
                    subcommand_config.model.init_args.model.init_args.ignore_snp = ignore_snp
                    print(f"Propagated ignore_snp={ignore_snp} to EyeWidthRegressor")

                # Pass to datamodule
                if hasattr(subcommand_config, 'data') and hasattr(subcommand_config.data, 'init_args'):
                    subcommand_config.data.init_args.ignore_snp = ignore_snp
                    print(f"Propagated ignore_snp={ignore_snp} to datamodule")

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