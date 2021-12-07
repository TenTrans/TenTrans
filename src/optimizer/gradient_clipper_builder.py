from typing import Optional, Callable
from torch import nn


def gradient_clipper_builder(config: dict) -> Optional[Callable]:
    """
    Define the function for gradient clipping as specified in configuration.
    If not specified, returns None.
    Current options:
        - "clip_grad_val": clip the gradients if they exceed this value,
            see `torch.nn.utils.clip_grad_value_`
        - "clip_grad_norm": clip the gradients if their norm
            exceeds this value, see `torch.nn.utils.clip_grad_norm_`
    :param config: dictionary with training configurations
    :return: clipping function (in-place) or None if no gradient clipping
    """

    clip_grad_fun = None
    if "clip_grad_val" in config.keys():
        clip_value = config["clip_grad_val"]

        def func(params):
            return nn.utils.clip_grad_value_(parameters=params, clip_value=clip_value)

        clip_grad_fun = func

    elif "clip_grad_norm" in config.keys():
        max_norm = config["clip_grad_norm"]

        def func(params):
            return nn.utils.clip_grad_norm_(parameters=params, max_norm=max_norm)

        clip_grad_fun = func

    if "clip_grad_val" in config.keys() and "clip_grad_norm" in config.keys():
        raise Exception(
            "You can only specify either \
            clip_grad_val or clip_grad_norm."
        )

    return clip_grad_fun
