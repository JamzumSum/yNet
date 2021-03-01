from datetime import timedelta

import pytorch_lightning as pl
from rich.progress import BarColumn, Progress, ProgressColumn, Text


class BarDictColumn(ProgressColumn):
    max_refresh = 0.5

    def time_estimated(self, task) -> str:
        remaining = task.time_remaining
        elapsed = task.elapsed
        if remaining is None or elapsed is None:
            return "-:--:--"
        estimated = timedelta(seconds=int(elapsed + remaining))
        return str(estimated)

    def time_elapsed(self, task) -> str:
        elapsed = task.elapsed
        if elapsed is None:
            return "-:--:--"
        elapsed = timedelta(seconds=int(elapsed))
        return str(elapsed)

    def bar_dict(self, bardic: dict) -> str:
        return ", ".join(f"{k}={v}" for k, v in bardic.items())

    def render(self, task):
        items = []
        items.append(f"{self.time_elapsed(task)}<{self.time_estimated(task)}")
        if task.speed is not None:
            items.append(f"{task.speed:>.2f}it/s")
        items.append(self.bar_dict(task.fields))
        return Text("[%s]" % ", ".join(items))


class RichProgressBar(pl.callbacks.ProgressBarBase):
    def __init__(self):
        super().__init__()
        self.pg = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.completed}/{task.total}",
            BarDictColumn(),
            transient=True,
            expand=True,
            refresh_per_second=2,
        )
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
        bardic = pl_module.get_progress_bar_dict()
        self.pg.update(self._test_id, completed=self.test_batch_idx, **bardic)
        if self.test_batch_idx >= self.total_test_batches:
            self.pg.stop_task(self._test_id)

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        if self._test_id is not None:
            self.pg.remove_task(self._test_id)
        self._test_id = self.pg.add_task(
            "epoch T%03d" % trainer.current_epoch, total=self.total_test_batches
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        bardic = pl_module.get_progress_bar_dict()
        self.pg.update(self._train_id, completed=self.train_batch_idx, **bardic)
        if self.train_batch_idx >= self.total_train_batches:
            self.pg.stop_task(self._train_id)

    def on_epoch_start(self, trainer, pl_module):
        super().on_epoch_start(trainer, pl_module)
        if self._train_id is not None:
            self.pg.remove_task(self._train_id)
        if self._val_id is not None:
            self.pg.remove_task(self._val_id)
        self._train_id = self.pg.add_task(
            "epoch T%03d" % trainer.current_epoch, total=self.total_train_batches
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        bardic = pl_module.get_progress_bar_dict()
        self.pg.update(self._val_id, completed=self.val_batch_idx, **bardic)
        if self.val_batch_idx >= self.total_val_batches:
            self.pg.stop_task(self._val_id)

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        self._val_id = self.pg.add_task(
            "epoch V%03d" % trainer.current_epoch, total=self.total_val_batches
        )

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self.pg.stop()

    def print(self, *args, **kwargs):
        self.pg.console.print(*args, **kwargs)

