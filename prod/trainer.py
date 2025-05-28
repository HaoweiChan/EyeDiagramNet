import sys
import torch
import warnings
from pathlib import Path
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities import disable_possible_user_warnings

class CustomLightningCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        if "predict" in sys.argv:
            kwargs['save_config_callback'] = None
        super().__init__(*args, **kwargs)

    def before_instantiate_classes(self):
        if self.subcommand == "predict":
            ckpt_config = self.config.predict.config[0]
            ckpt_dir = Path(str(ckpt_config)).parent / "checkpoints"
            ckpt_path = list(ckpt_dir.glob("*.ckpt"))[0]
            self.config.predict.model.init_args.ckpt_path = str(ckpt_path)

            scaler_path = list(Path(str(ckpt_config)).parent.glob("*.pth"))[0]
            self.config.predict.data.init_args.scaler_path = str(scaler_path)

    def instantiate_trainer(self, **kwargs):
        trainer = super().instantiate_trainer(**kwargs)
        config_path = self.config.predict.config[-1]
        config_dir = Path(config_path).parent
        for callback in trainer.callbacks:
            if hasattr(callback, "set_output_paths"):
                callback.set_output_paths(config_dir)

        return trainer

    # def after_instantiate_classes(self):
    #    config_paths = getattr(self.parser, "config_filenames", [])
    #    print(config_paths)
    #    print(self.trainer.callbacks)
    #    for callback in self.trainer.callbacks:
    #        if hasattr(callback, "set_output paths"):
    #            callback.set_output_paths(config_paths)
    #    raise

def cli_main():
    cli = CustomLightningCLI(datamodule_class=None)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    disable_possible_user_warnings()
    torch.set_float32_matmul_precision('medium')
    cli_main()