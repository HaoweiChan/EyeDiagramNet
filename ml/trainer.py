import sys
import yaml
import torch
import warnings
from pathlib import Path
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities import disable_possible_user_warnings


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
            elif subcommand in ["predict", "test"]:
                user_config_path = Path("configs/user/test_setting.yaml")
    
    # Load user settings if the file exists
    if user_config_path and user_config_path.exists():
        try:
            with open(user_config_path, 'r') as f:
                user_settings = yaml.safe_load(f)
            
            # For predict/test modes, add model_config_path to sys.argv if specified
            if len(sys.argv) > 1 and sys.argv[1] in ["predict", "test"]:
                if user_settings.get('model_config_path'):
                    model_config = user_settings['model_config_path']
                    # Add the model config as a second --config argument
                    # Find where to insert it (after main config)
                    config_index = -1
                    for i, arg in enumerate(sys.argv):
                        if arg == "--config" and i + 1 < len(sys.argv):
                            config_index = i + 2  # After the config file path
                            break
                    
                    if config_index > 0:
                        # Insert model config after main config
                        sys.argv.insert(config_index, "--config")
                        sys.argv.insert(config_index + 1, model_config)
                    else:
                        # If no --config found, add it
                        sys.argv.extend(["--config", model_config])
            
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
            
                # Handle predict and test modes - both need checkpoint and scaler loading
                config_attr = getattr(self.config, self.subcommand)
                
                # The model config should now be loaded as part of the configs
                # Find the model config from the loaded configs
                if hasattr(config_attr, 'config') and len(config_attr.config) > 1:
                    ckpt_config = config_attr.config[1]  # Second config should be the model config
                elif hasattr(config_attr, 'config') and len(config_attr.config) > 0:
                    ckpt_config = config_attr.config[0]  # Fallback to first config
                else:
                    print("Warning: No model config found in loaded configs. Please specify model_config_path in user settings or use --config argument.")
                    return
                
                if 'ckpt_config' in locals():
                    try:
                        # Find checkpoint in the model config directory
                        ckpt_dir = Path(str(ckpt_config)).parent / "checkpoints"
                        if ckpt_dir.exists():
                            ckpt_files = list(ckpt_dir.glob("*.ckpt"))
                            if ckpt_files:
                                ckpt_path = ckpt_files[0]  # Use first checkpoint found
                                config_attr.ckpt_path = str(ckpt_path)
                                print(f"Found checkpoint: {ckpt_path}")
                            else:
                                print(f"No .ckpt files found in {ckpt_dir}")
                        else:
                            print(f"Checkpoint directory not found: {ckpt_dir}")

                        # Find scaler in the model config directory  
                        scaler_files = list(Path(str(ckpt_config)).parent.glob("*.pth"))
                        if scaler_files:
                            scaler_path = scaler_files[0]  # Use first scaler found
                            config_attr.data.init_args.scaler_path = str(scaler_path)
                            print(f"Set scaler_path to: {scaler_path}")
                        else:
                            print(f"No .pth scaler files found in {Path(str(ckpt_config)).parent}")
                        
                    except Exception as e:
                        print(f"Warning: Failed to process checkpoint/scaler from config: {e}")
            
            # Pass ignore_snp flag from TraceEWModule to both nested model and datamodule
            if hasattr(self.config, self.subcommand) and hasattr(getattr(self.config, self.subcommand), 'model'):
                model_config = getattr(self.config, self.subcommand).model
                if hasattr(model_config, 'init_args') and hasattr(model_config.init_args, 'ignore_snp'):
                    ignore_snp = model_config.init_args.ignore_snp
                    
                    # Pass to nested model (EyeWidthRegressor)
                    if hasattr(model_config.init_args, 'model') and hasattr(model_config.init_args.model, 'init_args'):
                        model_config.init_args.model.init_args.ignore_snp = ignore_snp
                        print(f"Propagated ignore_snp={ignore_snp} from TraceEWModule to EyeWidthRegressor")
                    
                    # Pass to datamodule
                    data_config = getattr(self.config, self.subcommand).data
                    if hasattr(data_config, 'init_args'):
                        data_config.init_args.ignore_snp = ignore_snp
                        print(f"Propagated ignore_snp={ignore_snp} from TraceEWModule to datamodule")

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