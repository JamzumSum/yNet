import os

import yaml
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

from data.dataset.cacheset import DataMeta


class TestLogger(LightningLoggerBase):
    def __init__(self, save_dir, name='default', version=None, prefix=''):
        super().__init__()
        self.f = os.path.join(
            save_dir, name,
            prefix + ('test_raw.yml' if version is None else f'{version}.yml')
        )
        self._d = {}
        self.add = []

    @property
    def name(self):
        return self.__class__.__name__

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        return self._d

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @property
    def version(self):
        return

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()
        os.makedirs(os.path.dirname(self.f), exist_ok=True)
        with open(self.f, 'w') as f:
            yaml.safe_dump_all([self._d] + self.add, f)
        self._d.clear()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        self.save()

    @rank_zero_only
    def log_metrics(self, meta: DataMeta, **raw):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        raw['pid'] = meta.pid
        self._d[meta.pid] = raw

    def __del__(self):
        if self._d: self.save()