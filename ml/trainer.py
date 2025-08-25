import sys
import yaml
import torch
import warnings
from pathlib import Path
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities import disable_possible_user_warnings


class ConfigProcessor:
    """Handles configuration processing for predict and test modes."""
    
    def __init__(self, config, subcommand):
        self.config = config
        self.subcommand = subcommand
        self.sub_config = getattr(config, subcommand)
    
    def process_predict_config(self):
        """Process configuration for predict mode - auto-find checkpoint and scaler."""
        print("Processing predict config to auto-load scaler")
        
        # Find the version directory from the config paths 
        version_dir = None
        if hasattr(self.config.predict, 'config') and self.config.predict.config:
            print(f"Found {len(self.config.predict.config)} config paths")
            
            # Check each config path to find one in a version directory
            for i, config_path_obj in enumerate(self.config.predict.config):
                # Handle Path_fsr objects
                if hasattr(config_path_obj, '__fspath__'):
                    config_path = Path(config_path_obj.__fspath__())
                elif hasattr(config_path_obj, 'path'):
                    config_path = Path(str(config_path_obj.path))
                else:
                    config_path = Path(str(config_path_obj))
                
                print(f"Checking config path {i}: {config_path}")
                
                # Check if config file is in version_X/ directory
                if config_path.parent.name.startswith('version_'):
                    version_dir = config_path.parent
                    print(f"Found version directory from config: {version_dir}")
                    break
                elif len(config_path.parts) > 2 and config_path.parent.parent.name.startswith('version_'):
                    version_dir = config_path.parent.parent
                    print(f"Found version directory from config (grandparent): {version_dir}")
                    break
        
        if version_dir is None:
            print("Warning: Could not determine version directory for scaler auto-loading")
            return
        
        # Find checkpoint and scaler in version directory
        checkpoint_dir = version_dir / "checkpoints"
        if checkpoint_dir.exists():
            ckpt_path = next(reversed(sorted(checkpoint_dir.glob("*.ckpt"))))
            self.config.predict.ckpt_path = str(ckpt_path)
            print(f"Found checkpoint: {ckpt_path}")
        
        # Look for scaler.pth in version directory (not in checkpoints)
        scaler_files = list(version_dir.glob("*.pth"))
        print(f"Found {len(scaler_files)} .pth files in {version_dir}: {[f.name for f in scaler_files]}")
        
        if scaler_files:
            # Use scaler.pth if exists, otherwise the first .pth file
            scaler_path = None
            for f in scaler_files:
                if f.name == "scaler.pth":
                    scaler_path = f
                    break
            if scaler_path is None:
                scaler_path = scaler_files[0]
            
            self.config.predict.data.init_args.scaler_path = str(scaler_path)
            print(f"Set scaler_path to: {scaler_path}")
        else:
            print(f"No .pth scaler files found in {version_dir}")
    
    def process_test_config(self):
        """Process configuration for test mode - auto-find and set scaler path."""
        print("Processing test config to auto-load scaler")
        
        # Find the version directory from the config paths (handle multiple --config arguments)
        version_dir = None
        if hasattr(self.sub_config, 'config') and self.sub_config.config:
            print(f"Found {len(self.sub_config.config)} config paths")
            
            # Check each config path to find one in a version directory
            for i, config_path_obj in enumerate(self.sub_config.config):
                # Handle Path_fsr objects by accessing their path attribute or converting properly
                if hasattr(config_path_obj, '__fspath__'):
                    config_path = Path(config_path_obj.__fspath__())
                elif hasattr(config_path_obj, 'path'):
                    config_path = Path(str(config_path_obj.path))
                else:
                    config_path = Path(str(config_path_obj))
                
                print(f"Checking config path {i}: {config_path}")
                
                # Check if config file is in version_X/ directory
                if config_path.parent.name.startswith('version_'):
                    version_dir = config_path.parent
                    print(f"Found version directory from config: {version_dir}")
                    break
                # Also check grandparent in case config is in version_X/subdir/
                elif len(config_path.parts) > 2 and config_path.parent.parent.name.startswith('version_'):
                    version_dir = config_path.parent.parent
                    print(f"Found version directory from config (grandparent): {version_dir}")
                    break
        
        if version_dir is None:
            print("Warning: Could not determine version directory for scaler auto-loading")
            print("Available config paths checked:")
            if hasattr(self.sub_config, 'config') and self.sub_config.config:
                for config_path_obj in self.sub_config.config:
                    print(f"  - {config_path_obj}")
            return
        
        # Look for scaler.pth in the version directory (not in checkpoints subdirectory)
        scaler_files = list(version_dir.glob("*.pth"))
        print(f"Found {len(scaler_files)} .pth files in {version_dir}: {[f.name for f in scaler_files]}")
        
        if not scaler_files:
            print(f"No .pth scaler files found in {version_dir}")
            return
        
        # Use scaler.pth if exists, otherwise the first .pth file
        scaler_path = None
        for f in scaler_files:
            if f.name == "scaler.pth":
                scaler_path = f
                break
        if scaler_path is None:
            scaler_path = scaler_files[0]
        
        # Set scaler path in data configuration
        if hasattr(self.sub_config, 'data') and hasattr(self.sub_config.data, 'init_args'):
            self.sub_config.data.init_args.scaler_path = str(scaler_path)
            print(f"Set scaler_path to: {scaler_path}")
        else:
            print("Warning: Could not set scaler_path - data config structure not found")
    
    def process_user_config(self, user_config_path):
        """Process user configuration file and extract checkpoint/training config paths."""
        print(f"Processing user config: {user_config_path}")
        
        # Load user config
        with open(user_config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Extract checkpoint path
        ckpt_path = user_config.get('ckpt_path')
        if not ckpt_path:
            raise ValueError("User config must contain 'ckpt_path'")
        
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
        # Find training config in the checkpoint's version directory
        version_dir = ckpt_path.parent.parent  # checkpoints/model.ckpt -> version_X/
        training_config = version_dir / "config.yaml"
        
        if not training_config.exists():
            raise FileNotFoundError(f"Training config not found: {training_config}")
        
        print(f"Found checkpoint: {ckpt_path}")
        print(f"Found training config: {training_config}")
        
        # Return paths for CLI to use
        return str(training_config), user_config
    
    def apply_user_config_overrides(self, user_config):
        """Apply user config overrides to the loaded configuration."""
        print("Applying user config overrides")
        
        # Apply model checkpoint path
        if 'ckpt_path' in user_config:
            self.sub_config.ckpt_path = user_config['ckpt_path']
            print(f"Set checkpoint path: {user_config['ckpt_path']}")
        
        # Apply logging configuration
        if hasattr(self.sub_config, 'trainer') and hasattr(self.sub_config.trainer, 'logger'):
            if 'save_dir' in user_config:
                for logger in self.sub_config.trainer.logger:
                    if hasattr(logger, 'init_args') and hasattr(logger.init_args, 'save_dir'):
                        logger.init_args.save_dir = user_config['save_dir']
                        print(f"Set logger save_dir: {user_config['save_dir']}")
        
        # Apply prediction writer configuration
        if hasattr(self.sub_config, 'trainer') and hasattr(self.sub_config.trainer, 'callbacks'):
            if 'file_prefix' in user_config:
                for callback in self.sub_config.trainer.callbacks:
                    if hasattr(callback, 'init_args') and hasattr(callback.init_args, 'file_prefix'):
                        callback.init_args.file_prefix = user_config['file_prefix']
                        print(f"Set prediction file_prefix: {user_config['file_prefix']}")
        
        # Apply data configuration
        if hasattr(self.sub_config, 'data') and hasattr(self.sub_config.data, 'init_args'):
            data_args = self.sub_config.data.init_args
            
            # Data directories
            if 'data_dirs' in user_config:
                data_args.data_dirs = user_config['data_dirs']
                print(f"Set data_dirs: {len(user_config['data_dirs'])} directories")
            
            # SNP files
            if 'drv_snp' in user_config:
                data_args.drv_snp = user_config['drv_snp']
                print(f"Set drv_snp: {user_config['drv_snp']}")
            
            if 'odt_snp' in user_config:
                data_args.odt_snp = user_config['odt_snp']
                print(f"Set odt_snp: {user_config['odt_snp']}")
            
            # Boundary path
            if 'bound_path' in user_config:
                data_args.bound_path = user_config['bound_path']
                print(f"Set bound_path: {user_config['bound_path']}")
            
            # Processing options
            if 'ignore_snp' in user_config:
                data_args.ignore_snp = user_config['ignore_snp']
                print(f"Set ignore_snp: {user_config['ignore_snp']}")
    
    def process_config(self):
        """Process configuration for the current subcommand."""
        if self.subcommand == "predict":
            self.process_predict_config()
        elif self.subcommand == "test":
            self.process_test_config()

class CustomLightningCLI(LightningCLI):
    """Custom CLI to handle checkpoint and scaler path resolution for predictions and testing."""
    def __init__(self, *args, **kwargs):
        if 'predict' in sys.argv or 'test' in sys.argv:
            kwargs['save_config_callback'] = None
        
        # Handle user config preprocessing
        self.user_config = None
        self._preprocess_user_config()
        
        super().__init__(*args, **kwargs)
    
    def _preprocess_user_config(self):
        """Preprocess user config and modify sys.argv if needed."""
        # Check if --user_config is in arguments
        user_config_arg = None
        user_config_path = None
        
        for i, arg in enumerate(sys.argv):
            if arg == '--user_config' and i + 1 < len(sys.argv):
                user_config_arg = i
                user_config_path = sys.argv[i + 1]
                break
            elif arg.startswith('--user_config='):
                user_config_arg = i
                user_config_path = arg.split('=', 1)[1]
                break
        
        if user_config_path:
            print(f"Detected user config: {user_config_path}")
            
            # Determine subcommand
            subcommand = None
            for arg in sys.argv[1:]:
                if arg in ['predict', 'test', 'fit', 'validate']:
                    subcommand = arg
                    break
            
            if subcommand in ['predict', 'test']:
                try:
                    # Create temporary config processor to extract paths
                    class TempConfig:
                        def __init__(self):
                            pass
                    
                    temp_config = TempConfig()
                    setattr(temp_config, subcommand, TempConfig())
                    
                    processor = ConfigProcessor(temp_config, subcommand)
                    training_config_path, user_config = processor.process_user_config(user_config_path)
                    
                    # Store user config for later application
                    self.user_config = user_config
                    
                    # Insert training config into sys.argv right after --config arguments
                    config_insert_pos = None
                    for i, arg in enumerate(sys.argv):
                        if arg.startswith('--config'):
                            config_insert_pos = i + 2  # After --config and its value
                    
                    if config_insert_pos is None:
                        # No --config found, insert after subcommand
                        for i, arg in enumerate(sys.argv):
                            if arg == subcommand:
                                config_insert_pos = i + 1
                                break
                    
                    if config_insert_pos:
                        sys.argv.insert(config_insert_pos, training_config_path)
                        sys.argv.insert(config_insert_pos, '--config')
                        print(f"Added training config to command line: {training_config_path}")
                    
                    # Remove --user_config arguments from sys.argv
                    if user_config_arg is not None:
                        if sys.argv[user_config_arg].startswith('--user_config='):
                            # Single argument with =
                            sys.argv.pop(user_config_arg)
                        else:
                            # Two arguments: --user_config path
                            sys.argv.pop(user_config_arg)  # Remove --user_config
                            sys.argv.pop(user_config_arg)  # Remove path (indices shift after first pop)
                    
                except Exception as e:
                    print(f"Warning: Failed to process user config: {e}")
                    self.user_config = None

    def before_instantiate_classes(self):
        # Handle predict and test mode config processing
        if self.subcommand in ["predict", "test"]:
            # Apply user config overrides first
            if self.user_config:
                try:
                    processor = ConfigProcessor(self.config, self.subcommand)
                    processor.apply_user_config_overrides(self.user_config)
                except Exception as e:
                    print(f"Warning: Failed to apply user config overrides: {e}")
            
            # Then run standard config processing
            try:
                processor = ConfigProcessor(self.config, self.subcommand)
                processor.process_config()
            except (IndexError, StopIteration, FileNotFoundError) as e:
                print(f"Warning: Could not process config for {self.subcommand}: {e}")
        
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