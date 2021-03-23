import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


class TestLogger(LightningLoggerBase):
    def __init__(self, save_dir, name='default', prefix='', add: dict=None):
        super().__init__()
        self.add = add
        self.f = os.path.join(save_dir, name, prefix + 'test_raw.yml')
        self._d = {}

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
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        assert isinstance(params, dict)
        self.add = params.copy()

    @property
    def version(self):
        return

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()
        with open(self.f, 'w') as f:
            if self.add:
                yaml.safe_dump_all((self.add, self._d), f)
            else:
                yaml.safe_dump_all(({}, self._d), f)

    @rank_zero_only
    def log_metrics(self, meta, **raw):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        raw['meta'] = meta
        self._d[meta.pid] = raw
