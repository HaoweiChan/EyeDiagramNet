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
        # Test if torch.compile works with second-order gradients (GradNorm style)
        test_model = torch.nn.Linear(2, 1)
        test_input = torch.randn(1, 2, requires_grad=True)
        
        compiled_test = torch.compile(test_model, mode="default")
        output = compiled_test(test_input)
        loss1 = output.sum()
        
        # Test first-order gradients
        grad1 = torch.autograd.grad(loss1, test_input, create_graph=True)[0]
        
        # Test second-order gradients (this is what GradNorm does)
        loss2 = grad1.norm()
        loss2.backward()
        
        print("torch.compile test with second-order gradients passed - compilation will be attempted")
        return True
    except Exception as e:
        print(f"torch.compile test failed (likely due to second-order gradients): {e}")
        print("Your model uses GradNormLossBalancer which requires second-order gradients")
        print("torch.compile with aot_autograd doesn't support this - disabling compilation")
        
        # Disable torch.compile more aggressively
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.disable = True
        
        # Set environment variable to disable compilation
        import os
        os.environ['TORCH_COMPILE_DISABLE'] = '1'
        
        return False

def cli_main():
    setup_torch_compile_fallback()
    cli = CustomLightningCLI(datamodule_class=None)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    disable_possible_user_warnings()
    torch.set_float32_matmul_precision('medium')
    cli_main()