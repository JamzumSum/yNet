import pytorch_lightning as pl
import rich
from rich.progress import Progress


class RichProgressBar(pl.callbacks.ProgressBarBase):
    def __init__(self):
        super().__init__()
        self.pg = Progress(transient=True, expand=True, refresh_per_second=2)
        self._train_id = self._val_id = self._test_id = None

    def disable(self):
        self.pg.disable = True

    def enable(self):
        self.pg.disable = False

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.pg.start()


    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        self.pg.update(self._test_id, completed=self.test_batch_idx)

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        if self._test_id is not None:
            self.pg.remove_task(self._test_id)
        self._test_id = self.pg.add_task(
            "test %03d" % trainer.current_epoch, total=self.total_test_batches
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        self.pg.update(self._train_id, completed=self.train_batch_idx)

    def on_epoch_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        if self._train_id is not None:
            self.pg.remove_task(self._train_id)
        self._train_id = self.pg.add_task(
            "train %03d" % trainer.current_epoch, total=self.total_train_batches
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        self.pg.update(self._val_id, completed=self.val_batch_idx)

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        if self._val_id is not None:
            self.pg.remove_task(self._val_id)
        self._val_id = self.pg.add_task(
            "validate %03d" % trainer.current_epoch, total=self.total_val_batches
        )

    def __del__(self):
        self.pg.stop()

    def print(self, *args, **kwargs):
        rich.print(*args, **kwargs)
