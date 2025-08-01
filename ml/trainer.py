import sys
import torch
import warnings
from pathlib import Path
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities import disable_possible_user_warnings

class CustomLightningCLI(LightningCLI):
    """Custom CLI to handle checkpoint and scaler path resolution for predictions."""
    def __init__(self, *args, **kwargs):
        if 'predict' in sys.argv:
            kwargs['save_config_callback'] = None
        super().__init__(*args, **kwargs)

    def before_instantiate_classes(self):
        if self.subcommand == "predict":
            try:
                ckpt_config = self.config.predict.config[0]
                ckpt_dir = Path(str(ckpt_config)).parent
                ckpt_path = next(reversed(sorted(ckpt_dir.glob("*.ckpt"))))
                self.config.predict.ckpt_path = str(ckpt_path)
                scaler_path = next(ckpt_dir.glob("*.pth"))
                self.config.predict.data.init_args.scaler_path = str(scaler_path)
                print(f"Automatically using checkpoint: {ckpt_path} and scaler: {scaler_path}")
            except (IndexError, StopIteration, FileNotFoundError):
                print("Warning: Could not automatically find checkpoint or scaler for prediction.")
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