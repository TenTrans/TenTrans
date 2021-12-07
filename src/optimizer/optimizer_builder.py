from torch.optim import optimizer
from typing import Generator
import torch
import math


# https://blog.csdn.net/andyjkt/article/details/107389515
def optimizer_builder(config: dict, parameters: Generator) -> optimizer:

    assert "optimizer" in config
    optimizer_name = config.get("optimizer", "sgd").lower()
    learning_rate = config.get("learning_rate", 3.0e-4)
    weight_decay = config.get("weight_decay", 0.01)

    if optimizer_name == "adam":
        adam_betas = config.get("adam_betas", (0.9, 0.999))
        eps = config.get("eps", 1e-8)
        optimizer = Adam(
            parameters,
            weight_decay=weight_decay,
            lr=learning_rate,
            betas=adam_betas,
            eps=eps,
        )
    elif optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(parameters,
                                        weight_decay=weight_decay,
                                        lr=learning_rate)
    elif optimizer_name == "adadelta":
        optimizer = torch.optim.Adadelta(parameters,
                                         weight_decay=weight_decay,
                                         lr=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(parameters,
                                        weight_decay=weight_decay,
                                        lr=learning_rate)
    elif optimizer_name == "sgd":
        # default
        optimizer = torch.optim.SGD(parameters,
                                    weight_decay=weight_decay,
                                    lr=learning_rate)
    else:
        raise Exception("Invalid optimizer. Valid options: 'adam', "
                        "'adagrad', 'adadelta', 'rmsprop', 'sgd'.")
    return optimizer


class Adam(torch.optim.Optimizer):
    """
    Same as https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py,
    without amsgrad, with step in a tensor, and 
    states initialization  in __init__. It was important to 
    add `.item()` in `state['step'].item()`.
    see https://github.com/facebookresearch/XLM/blob/master/src/optim.py 
    for more details
    This implemenation is better than the naive one.
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(
                betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0  # torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

    def __setstate__(self, state):
        super().__setstate__(state)

    def set_lr(self, lr):
        for param_group in self.param_groups:
            param_group["lr"] = lr

    def step(self, closure=None):
        """
        Step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, \
                            please consider SparseAdam instead")

                state = self.state[p]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # if group['weight_decay'] != 0:
                #     grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                # denom = exp_avg_sq.sqrt().clamp_(min=group['eps'])

                bias_correction1 = 1 - beta1**state["step"]  # .item()
                bias_correction2 = 1 - beta2**state["step"]  # .item()
                step_size = group["lr"] * math.sqrt(
                    bias_correction2) / bias_correction1

                if group["weight_decay"] != 0:
                    p.data.add_(-group["weight_decay"] * group["lr"], p.data)

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
