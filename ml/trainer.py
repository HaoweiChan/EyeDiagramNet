import sys
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
        ckpt_config = self.config.predict.config[0]
        ckpt_dir = Path(str(ckpt_config)).parent
        ckpt_path = next(reversed(sorted(ckpt_dir.glob("*.ckpt"))))
        self.config.predict.ckpt_path = str(ckpt_path)
        scaler_path = next(ckpt_dir.glob("*.pth"))
        self.config.predict.data.init_args.scaler_path = str(scaler_path)
        print(f"Automatically using checkpoint: {ckpt_path} and scaler: {scaler_path}")
    
    def process_test_config(self):
        """Process configuration for test mode - auto-find and set scaler path."""
        print("Processing test config to auto-load scaler")
        
        # Find the version directory from the config path
        version_dir = None
        if hasattr(self.config, 'config') and self.config.config:
            config_path = Path(str(self.config.config[0]))
            # Config file should be in version_X/ directory
            if config_path.parent.name.startswith('version_'):
                version_dir = config_path.parent
                print(f"Found version directory from config: {version_dir}")
        
        if version_dir is None:
            print("Warning: Could not determine version directory for scaler auto-loading")
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
        super().__init__(*args, **kwargs)

    def before_instantiate_classes(self):
        # Handle predict and test mode config processing
        if self.subcommand in ["predict", "test"]:
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