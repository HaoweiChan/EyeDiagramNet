import sys
import torch
import warnings
from pathlib import Path
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities import disable_possible_user_warnings

class CustomLightningCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        if 'predict' in sys.argv:
            kwargs['save_config_callback'] = None
        super().__init__(*args, **kwargs)

    def before_instantiate_classes(self):
        if self.subcommand == "predict":
            ckpt_config = self.config.predict.config[0]
            ckpt_dir = Path(str(ckpt_config)).parent / "checkpoints"
            ckpt_path = list(ckpt_dir.glob("*.ckpt"))[0]
            self.config.predict.ckpt_path = str(ckpt_path)

            scaler_path = list(Path(str(ckpt_config)).parent.glob("*.pth"))[0]
            self.config.predict.data.init_args.scaler_path = str(scaler_path)

def setup_torch_compile_fallback():
    """Setup torch.compile fallback to eager mode on compilation failures"""
    try:
        # Test if torch.compile is working properly
        test_model = torch.nn.Linear(2, 1)
        compiled_test = torch.compile(test_model, mode="default")
        # If this succeeds, torch.compile should work
        return True
    except Exception as e:
        print(f"torch.compile test failed: {e}")
        print("Disabling torch.compile globally - falling back to eager mode")
        torch._dynamo.config.suppress_errors = True
        return False

def cli_main():
    setup_torch_compile_fallback()
    cli = CustomLightningCLI(datamodule_class=None)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    disable_possible_user_warnings()
    torch.set_float32_matmul_precision('medium')
    cli_main()