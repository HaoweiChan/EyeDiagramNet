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
        """Process configuration for test mode - load training config from checkpoint."""
        # Get ckpt_path from model.init_args.ckpt_path
        ckpt_path = None
        if (hasattr(self.sub_config, 'model') and 
            hasattr(self.sub_config.model, 'init_args') and 
            hasattr(self.sub_config.model.init_args, 'ckpt_path')):
            ckpt_path = self.sub_config.model.init_args.ckpt_path
        
        if not ckpt_path:
            print("No checkpoint path found in test config")
            return
            
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print(f"Loading training config from checkpoint: {ckpt_path}")
        
        if "hyper_parameters" not in ckpt:
            print("Warning: No hyper_parameters found in checkpoint")
            return
            
        train_hparams = ckpt["hyper_parameters"]
        print(f"Found {len(train_hparams)} hyperparameters in checkpoint")
        print(f"Hyperparameter keys: {list(train_hparams.keys()) if hasattr(train_hparams, 'keys') else 'Not dict-like'}")
        
        self._merge_training_config(train_hparams)
        print("Successfully merged training config with test config.")
    
    def _merge_training_config(self, train_hparams):
        """Merge training hyperparameters with test config."""
        # Preserve test-specific overrides first
        test_overrides = self._get_test_overrides()
        
        # Convert train_hparams to dict if it's not already
        if hasattr(train_hparams, '__dict__'):
            hparams_dict = vars(train_hparams)
        elif hasattr(train_hparams, 'items'):
            hparams_dict = dict(train_hparams.items())
        else:
            hparams_dict = dict(train_hparams)
        
        # Merge all training hyperparameters into model.init_args
        for key, value in hparams_dict.items():
            # Skip test-specific overrides to preserve them
            if key not in test_overrides:
                # Ensure the attribute exists in init_args or create it
                if not hasattr(self.sub_config.model.init_args, key):
                    setattr(self.sub_config.model.init_args, key, None)
                setattr(self.sub_config.model.init_args, key, value)
        
        # Restore test-specific overrides (these take precedence)
        for key, value in test_overrides.items():
            setattr(self.sub_config.model.init_args, key, value)
    
    def _get_test_overrides(self):
        """Get test-specific configuration overrides that should be preserved."""
        test_overrides = {}
        # Common test-specific parameters that should override training config
        override_keys = [
            'ckpt_path',                # Checkpoint path for loading weights
            'use_laplace_on_fit_end',   # Disable Laplace for testing
            'compile_model',            # Disable compilation for testing
            'strict'                    # Checkpoint loading strictness
        ]
        
        for key in override_keys:
            if hasattr(self.sub_config.model.init_args, key):
                value = getattr(self.sub_config.model.init_args, key)
                # Only preserve non-None values
                if value is not None:
                    test_overrides[key] = value
        
        return test_overrides

class CustomLightningCLI(LightningCLI):
    """Custom CLI to handle checkpoint and scaler path resolution for predictions."""
    def __init__(self, *args, **kwargs):
        if 'predict' in sys.argv:
            kwargs['save_config_callback'] = None
        super().__init__(*args, **kwargs)

    def before_instantiate_classes(self):
        if self.subcommand in ["predict", "test"]:
            try:
                processor = ConfigProcessor(self.config, self.subcommand)
                
                if self.subcommand == "predict":
                    processor.process_predict_config()
                elif self.subcommand == "test":
                    processor.process_test_config()
                    
            except (IndexError, StopIteration, FileNotFoundError) as e:
                print(f"Warning: Could not process config for {self.subcommand}: {e}")
                pass
        
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