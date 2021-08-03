from torch.optim import optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, StepLR, ExponentialLR
from enum import Enum
from typing import Optional
import torch


def lr_sheduler_builder(
        config: dict,
        optimizer: optimizer,
        mode: str = "") -> (Optional[_LRScheduler], Optional[str]):

    scheduler, scheduler_step_at_str = None, None

    class scheduler_step_at(Enum):
        validation = 1
        step = 2
        epoch = 3

    if "scheduling" in config.keys() and \
            config["scheduling"]:
        # See explanation in https://zhuanlan.zhihu.com/p/69411064
        if config["scheduling"].lower() == "plateau":
            # learning rate scheduler
            scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                          mode=mode,
                                          verbose=False,
                                          threshold_mode='abs',
                                          factor=config.get(
                                              "decrease_factor", 0.1),
                                          patience=config.get("patience", 10))
            # scheduler step is executed after every validation
            scheduler_step_at = "validation"
        elif config["scheduling"].lower() == "exponential":
            # see explanation in https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ExponentialLR
            scheduler = ExponentialLR(optimizer=optimizer,
                                      gamma=config.get("decrease_factor",
                                                       0.99))
            # scheduler step is executed after every epoch
            scheduler_step_at_str = scheduler_step_at.epoch.name
        elif config["scheduling"].lower() == "noam":
            factor = config.get("learning_rate_factor", 1)
            warmup = config.get("learning_rate_warmup", 4000)
            scheduler = NoamScheduler(
                model_size=config["model"]["encoder"]["hidden_size"],
                factor=factor,
                warmup=warmup,
                optimizer=optimizer)

            scheduler_step_at_str = scheduler_step_at.step.name
        elif config["scheduling"].lower() == "warmupexponentialdecay":
            lr = config.get("learning_rate", 1.0e-5)
            decay_rate = config.get("learning_rate_decay", 0.5)
            warmup = config.get("learning_rate_warmup", 4000)
            warmup_init_lr = config.get("learning_warmup_init_lr", 1e-7)
            scheduler = WarmupExponentialDecayScheduler(
                lr=lr,
                exp_decay=decay_rate,
                warmup=warmup,
                optimizer=optimizer,
                warmup_init_lr=warmup_init_lr)
            scheduler_step_at_str = scheduler_step_at.step.name
    else:
        scheduler = BaseScheduler()
        scheduler_step_at_str = scheduler_step_at.step.name
    return scheduler, scheduler_step_at_str


class BaseScheduler:
    def __init__(self):
        pass

    def step(self):
        pass

    def state_dict(self):
        return {}


class NoamScheduler:
    """
    The Noam learning rate scheduler used in "Attention is all you need"
    See Eq. 3 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self,
                 model_size: int,
                 optimizer: torch.optim.Optimizer,
                 factor: float = 1,
                 warmup: int = 4000):
        """
        Warm-up, followed by learning rate decay.
        :param model_size:
        :param optimizer:
        :param factor: decay factor
        :param warmup: number of warmup steps
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.compute_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

    def compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def get_lr(self):
        return self._rate

    #pylint: disable=no-self-use
    def state_dict(self):
        return None


class WarmupExponentialDecayScheduler:
    """
    A learning rate scheduler similar to Noam, but modified:
    Keep the warm up period but make it so that the decay rate can be tuneable.
    The decay is exponential up to a given minimum rate.

    See explanation in https://www.zhihu.com/column/p/29421235
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 lr: float = 1.0e-3,
                 warmup: float = 4000,
                 exp_decay: float = 0.5,
                 warmup_init_lr: float = 1.0e-7,
                 min_lr: float = 1.0e-9):
        """
        Warm-up, followed by exponential learning rate decay.
        :param peak_rate: maximum learning rate at peak after warmup
        :param optimizer:
        :param decay_length: decay length after warmup
        :param decay_rate: decay rate after warmup
        :param warmup: number of warmup steps
        :param min_rate: minimum learning rate
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.lr = lr
        self._rate = 0
        self.min_lr = min_lr
        self.exp_decay = exp_decay
        self.decay_rate = self.lr * warmup**exp_decay
        self.warmup_init_lr = warmup_init_lr
        self.lr_step = (self.lr - self.warmup_init_lr) / warmup
        self.optimizer.set_lr(self.warmup_init_lr)

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.compute_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

    def compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        warmup = self.warmup
        if step < warmup:
            return max(self.warmup_init_lr + step * self.lr_step, self.min_lr)
        else:
            return max(self.decay_rate * (step**-self.exp_decay), self.min_lr)

    def get_step(self):
        return self._step

    #pylint: disable=no-self-use
    def state_dict(self):
        return {'step': self._step, 'rate': self._rate}

    def load_state_dict(self, data):
        self._step = data['step']
        self._rate = data['rate']
        self.optimizer.set_lr(self._rate)
