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
    
    def _find_checkpoint_directory(self):
        """Find the checkpoint directory from various config sources."""
        ckpt_dir = None
        
        if self.subcommand == "predict":
            # For predict mode, get from config array
            ckpt_config = self.config.predict.config[0]
            ckpt_dir = Path(str(ckpt_config)).parent
        elif self.subcommand == "test":
            # For test mode, look for checkpoint path in model config or use config directory
            if hasattr(self.sub_config, 'model') and hasattr(self.sub_config.model, 'init_args'):
                if hasattr(self.sub_config.model.init_args, 'ckpt_path'):
                    ckpt_path = self.sub_config.model.init_args.ckpt_path
                    ckpt_dir = Path(str(ckpt_path)).parent
            
            # Fallback: try to find from the main config file path if available
            if ckpt_dir is None and hasattr(self.config, 'config') and self.config.config:
                config_path = Path(str(self.config.config[0]))
                # Check if this looks like a checkpoint directory structure
                if config_path.parent.name.startswith('version_'):
                    ckpt_dir = config_path.parent
        
        return ckpt_dir
    
    def _auto_load_scaler(self, ckpt_dir):
        """Automatically find and set scaler path from checkpoint directory."""
        try:
            scaler_path = next(ckpt_dir.glob("scaler.pth"))
            if hasattr(self.sub_config, 'data') and hasattr(self.sub_config.data, 'init_args'):
                self.sub_config.data.init_args.scaler_path = str(scaler_path)
                print(f"Automatically using scaler: {scaler_path}")
                return True
        except StopIteration:
            try:
                # Fallback to any .pth file
                scaler_path = next(ckpt_dir.glob("*.pth"))
                if hasattr(self.sub_config, 'data') and hasattr(self.sub_config.data, 'init_args'):
                    self.sub_config.data.init_args.scaler_path = str(scaler_path)
                    print(f"Automatically using scaler: {scaler_path}")
                    return True
            except StopIteration:
                print(f"Warning: No scaler file found in {ckpt_dir}")
                return False
        return False
    
    def process_predict_config(self):
        """Process configuration for predict mode - auto-find checkpoint and scaler."""
        ckpt_config = self.config.predict.config[0]
        ckpt_dir = Path(str(ckpt_config)).parent
        ckpt_path = next(reversed(sorted(ckpt_dir.glob("*.ckpt"))))
        self.config.predict.ckpt_path = str(ckpt_path)
        scaler_path = next(ckpt_dir.glob("*.pth"))
        self.config.predict.data.init_args.scaler_path = str(scaler_path)
        print(f"Automatically using checkpoint: {ckpt_path} and scaler: {scaler_path}")
    
    def process_config(self):
        """Process configuration for the current subcommand."""
        if self.subcommand == "predict":
            self.process_predict_config()
        elif self.subcommand == "test":
            self.process_test_config()
    
    def process_test_config(self):
        """Process configuration for test mode - auto-find and set scaler path."""
        ckpt_dir = self._find_checkpoint_directory()
        
        if ckpt_dir is None:
            print("Warning: Could not determine checkpoint directory for auto-loading scaler")
            return
        
        print(f"Found checkpoint directory: {ckpt_dir}")
        success = self._auto_load_scaler(ckpt_dir)

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