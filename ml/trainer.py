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
    
    print(f"DEBUG: Creating temp config for {subcommand} with user settings")
    
    # Inject user settings into the config
    if user_settings.get('data_dirs'):
        data_dirs = user_settings['data_dirs']
        # Convert list format to dictionary format
        if isinstance(data_dirs, list):
            data_dirs_dict = {}
            for path in data_dirs:
                key = Path(path).name
                data_dirs_dict[key] = path
            print(f"DEBUG: Converted data_dirs to dict: {data_dirs_dict}")
        else:
            data_dirs_dict = data_dirs
            print(f"DEBUG: Using data_dirs as dict: {data_dirs_dict}")
        
        config['data']['init_args']['data_dirs'] = data_dirs_dict
    
    if user_settings.get('label_dir'):
        config['data']['init_args']['label_dir'] = user_settings['label_dir']
        print(f"DEBUG: Set label_dir to: {user_settings['label_dir']}")
    
    # Handle scaler path from checkpoint directory
    if user_settings.get('ckpt_path'):
        ckpt_path = Path(user_settings['ckpt_path'])
        version_dir = ckpt_path.parent.parent
        scaler_files = list(version_dir.glob("*.pth"))
        print(f"DEBUG: Looking for scaler files in {version_dir}, found: {scaler_files}")
        if scaler_files:
            scaler_path = str(scaler_files[0])
            config['data']['init_args']['scaler_path'] = scaler_path
            print(f"DEBUG: Set scaler_path to: {scaler_path}")
        else:
            print(f"DEBUG: Warning - No .pth scaler files found in {version_dir}")
    
    # Create temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix=f'temp_{subcommand}_')
    try:
        with os.fdopen(temp_fd, 'w') as f:
            yaml.safe_dump(config, f)
        print(f"DEBUG: Created temporary config at: {temp_path}")
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
    
    print("DEBUG: Starting preprocess_user_config")
    
    # Check if we have a user_config argument
    user_config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--user_config" and i + 1 < len(sys.argv):
            user_config_path = Path(sys.argv[i + 1])
            print(f"DEBUG: Found user_config arg: {user_config_path}")
            break
    
    # If no user_config specified, try default paths based on subcommand
    if not user_config_path:
        if len(sys.argv) > 1:
            subcommand = sys.argv[1]
            print(f"DEBUG: Subcommand detected: {subcommand}")
            if subcommand == "fit":
                user_config_path = Path("configs/user/train_setting.yaml")
            elif subcommand == "predict":
                user_config_path = Path("configs/user/infer_setting.yaml")
            elif subcommand == "test":
                user_config_path = Path("configs/user/test_setting.yaml")
            print(f"DEBUG: Using default user config path: {user_config_path}")
    
    # Load user settings if the file exists
    if user_config_path and user_config_path.exists():
        print(f"DEBUG: Loading user config from: {user_config_path}")
        try:
            with open(user_config_path, 'r') as f:
                user_settings = yaml.safe_load(f)
            print(f"DEBUG: Successfully loaded user settings: {user_settings}")
            
            # For predict/test modes, find training config from ckpt_path and inject it
            if len(sys.argv) > 1 and sys.argv[1] in ["predict", "test"]:
                if user_settings.get('ckpt_path'):
                    ckpt_path = Path(user_settings['ckpt_path'])
                    version_dir = ckpt_path.parent.parent
                    print(f"DEBUG: Looking for training config in: {version_dir}")

                    # Find training config
                    training_config_path = version_dir / 'config.yaml'
                    if not training_config_path.exists():
                        possible_configs = list(version_dir.glob("*.yaml"))
                        print(f"DEBUG: No config.yaml found, possible configs: {possible_configs}")
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
                    
                    print(f"DEBUG: Using training config: {training_config_path}")
                    
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
                        print(f"DEBUG: Inserted training config at position {config_index}")
                    else:
                        sys.argv.extend(["--config", str(training_config_path)])
                        print("DEBUG: Added training config to end of sys.argv")
            
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
                    print(f"DEBUG: Found main config: {main_config_path}")
                    # Create temporary config with user settings injected
                    temp_config_path = create_temp_config_with_user_settings(
                        main_config_path, user_settings, subcommand
                    )
                    
                    # Replace the main config path in sys.argv
                    sys.argv[main_config_index] = temp_config_path
                    print(f"DEBUG: Replaced config in sys.argv with temp config: {temp_config_path}")
            
            return user_settings
        except Exception as e:
            print(f"DEBUG: Error loading user config {user_config_path}: {e}")
            return None
    else:
        print(f"DEBUG: User config not found at: {user_config_path}")
    
    print("DEBUG: No user settings loaded")
    return None


class CustomLightningCLI(LightningCLI):
    """Custom CLI to handle checkpoint and scaler path resolution for predictions and testing."""
    def __init__(self, *args, **kwargs):
        if 'predict' in sys.argv or 'test' in sys.argv:
            kwargs['save_config_callback'] = None
        
        # Preprocess user config before Lightning CLI initialization
        self.user_settings = preprocess_user_config()
        print(f"DEBUG: CustomLightningCLI initialized with user_settings: {self.user_settings}")
        super().__init__(*args, **kwargs)
        print("DEBUG: Lightning CLI super().__init__() completed")
        
        # Apply user settings immediately after Lightning CLI initialization
        try:
            print("DEBUG: About to call _apply_user_settings_to_config()")
            self._apply_user_settings_to_config()
            print("DEBUG: _apply_user_settings_to_config() completed successfully")
        except Exception as e:
            print(f"DEBUG: Error in _apply_user_settings_to_config(): {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
    
    def add_arguments_to_parser(self, parser):
        # Add user config argument for all modes
        parser.add_argument("--user_config", type=str, help="Path to user configuration file")
    
    def _apply_user_settings_to_config(self):
        """Apply user settings to config immediately after Lightning CLI initialization."""
        print(f"DEBUG: _apply_user_settings_to_config called for subcommand: {self.subcommand}")
        print(f"DEBUG: user_settings available: {self.user_settings is not None}")
        
        if self.subcommand in ["predict", "test"] and self.user_settings:
            print("DEBUG: Applying user settings to configuration")
            config_attr = getattr(self.config, self.subcommand)
            print(f"DEBUG: Got config_attr for {self.subcommand}")
            print(f"DEBUG: config_attr.data: {config_attr.data}")
            print(f"DEBUG: config_attr.data.init_args: {config_attr.data.init_args}")
            print(f"DEBUG: Current data_dirs value: {getattr(config_attr.data.init_args, 'data_dirs', 'NOT_FOUND')}")
            
            # Apply user settings to data configuration if specified
            if self.user_settings.get('data_dirs'):
                data_dirs = self.user_settings['data_dirs']
                # Convert list format to dictionary format (dataloader expects dict[str, str])
                if isinstance(data_dirs, list):
                    # Convert list to dictionary with auto-generated keys
                    data_dirs_dict = {}
                    for i, path in enumerate(data_dirs):
                        # Extract basename from path for the key
                        key = Path(path).name
                        data_dirs_dict[key] = path
                    print(f"DEBUG: Converting data_dirs from list to dict: {data_dirs_dict}")
                else:
                    data_dirs_dict = data_dirs
                    print(f"DEBUG: Using data_dirs as dict: {data_dirs_dict}")
                
                config_attr.data.init_args.data_dirs = data_dirs_dict
                print(f"DEBUG: Set data_dirs to: {data_dirs_dict}")
                # Verify the assignment worked
                actual_value = getattr(config_attr.data.init_args, 'data_dirs', 'NOT_FOUND')
                print(f"DEBUG: Verification - data_dirs is now: {actual_value}")
                print(f"DEBUG: Assignment successful: {actual_value == data_dirs_dict}")
                
            if self.user_settings.get('label_dir'):
                config_attr.data.init_args.label_dir = self.user_settings['label_dir']
                print(f"DEBUG: Set label_dir to: {self.user_settings['label_dir']}")
                
            if self.user_settings.get('batch_size'):
                config_attr.data.init_args.batch_size = self.user_settings['batch_size']
                print(f"DEBUG: Set batch_size to: {self.user_settings['batch_size']}")
                
            if 'drv_snp' in self.user_settings:
                config_attr.data.init_args.drv_snp = self.user_settings['drv_snp']
                print(f"DEBUG: Set drv_snp to: {self.user_settings['drv_snp']}")
                
            if 'odt_snp' in self.user_settings:
                config_attr.data.init_args.odt_snp = self.user_settings['odt_snp']
                print(f"DEBUG: Set odt_snp to: {self.user_settings['odt_snp']}")
                
            if 'bound_path' in self.user_settings:
                config_attr.data.init_args.bound_path = self.user_settings['bound_path']
                print(f"DEBUG: Set bound_path to: {self.user_settings['bound_path']}")
                
            if self.user_settings.get('ignore_snp') is not None:
                config_attr.data.init_args.ignore_snp = self.user_settings['ignore_snp']
                print(f"DEBUG: Set ignore_snp to: {self.user_settings['ignore_snp']}")
            
            # Handle checkpoint and scaler paths
            if self.user_settings.get('ckpt_path'):
                ckpt_path_str = self.user_settings['ckpt_path']
                config_attr.ckpt_path = ckpt_path_str
                print(f"DEBUG: Set ckpt_path from user settings: {ckpt_path_str}")
                
                # Find scaler in the model version directory, derived from ckpt_path
                version_dir = Path(ckpt_path_str).parent.parent
                scaler_files = list(version_dir.glob("*.pth"))
                print(f"DEBUG: Looking for scaler files in {version_dir}, found: {scaler_files}")
                if scaler_files:
                    scaler_path = scaler_files[0]  # Use first scaler found
                    config_attr.data.init_args.scaler_path = str(scaler_path)
                    print(f"DEBUG: Set scaler_path to: {scaler_path}")
                else:
                    print(f"DEBUG: Warning - No .pth scaler files found in {version_dir}")
            else:
                print("DEBUG: Warning - ckpt_path not found in user settings")
        else:
            print("DEBUG: No user settings to apply or not in predict/test mode")

    def before_instantiate_classes(self):
        print(f"DEBUG: before_instantiate_classes called for subcommand: {self.subcommand}")
        
        # Pass ignore_snp flag from model to datamodule for all subcommands
        if self.subcommand and hasattr(self.config, self.subcommand):
            subcommand_config = getattr(self.config, self.subcommand)
            if hasattr(subcommand_config, 'model') and hasattr(subcommand_config.model, 'init_args') and \
               hasattr(subcommand_config.model.init_args, 'ignore_snp'):
                
                ignore_snp = subcommand_config.model.init_args.ignore_snp
                print(f"DEBUG: Found ignore_snp={ignore_snp} in model config")
                
                # Pass to nested model (EyeWidthRegressor) if it exists
                if hasattr(subcommand_config.model.init_args, 'model') and hasattr(subcommand_config.model.init_args.model, 'init_args'):
                    subcommand_config.model.init_args.model.init_args.ignore_snp = ignore_snp
                    print(f"DEBUG: Propagated ignore_snp={ignore_snp} to EyeWidthRegressor")

                # Pass to datamodule
                if hasattr(subcommand_config, 'data') and hasattr(subcommand_config.data, 'init_args'):
                    subcommand_config.data.init_args.ignore_snp = ignore_snp
                    print(f"DEBUG: Propagated ignore_snp={ignore_snp} to datamodule")


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