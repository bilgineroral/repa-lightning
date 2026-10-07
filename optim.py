import math
import re

import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from modules import LayerScale

# NEPA's weight decay exclusion (EnhancedTrainer.get_decay_parameter_names in original_nepa)
NO_DECAY_PATTERNS = [r"bias", r"layernorm", r"rmsnorm", r"layer_scale", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]

def get_parameter_names(model, forbidden_layer_types, forbidden_layer_names=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer
    (transformers.trainer_pt_utils.get_parameter_names).
    """
    forbidden_layer_patterns = (
        [re.compile(pattern) for pattern in forbidden_layer_names] if forbidden_layer_names is not None else []
    )
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(forbidden_layer_types))
            and not any(pattern.search(f"{name}.{n}".lower()) for pattern in forbidden_layer_patterns)
        ]
    # Add model specific parameters that are not in any child
    result += [
        k for k in model._parameters if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
    ]

    return result

def get_decay_parameter_names(model: nn.Module) -> list[str]:
    """ NEPA's rule. NEPA's layer scales are excluded by the name pattern "layer_scale", ours by type """
    return get_parameter_names(model, [nn.LayerNorm, LayerScale], NO_DECAY_PATTERNS)

def hf_accumulation_scale(batch_idx: int, num_batches: int, accumulate_grad_batches: int) -> float:
    """
    Loss multiplier matching HF's Trainer, which averages the last, partial gradient accumulation of an epoch
    over its own number of batches (Lightning divides every batch by accumulate_grad_batches).
    """
    remainder = num_batches % accumulate_grad_batches
    if remainder and batch_idx >= num_batches - remainder:
        return accumulate_grad_batches / remainder
    return 1.0

def get_llrd_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    NEPA's layer-wise lr decay schedule ("llrd_cosine_warmup" in original_nepa/schedulers.py).
    Each param group is scaled by llrd ** llrd_scale; groups with a smaller factor warm up for longer
    (a factor of 1 warms up in one step, a factor near 0 over num_warmup_steps), then decay with a cosine.
    """
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be > 0")

    num_warmup_steps = int(num_warmup_steps)
    num_training_steps = int(num_training_steps)

    def lr_lambda(current_step: int, llrd: float, llrd_scale: float) -> float:
        if current_step < 0:
            return 0.0

        factor = float(llrd) ** float(llrd_scale)
        if factor < 0.0:
            factor = 0.0

        if num_warmup_steps > 0:
            raw_warmup = int(round(num_warmup_steps * min(factor, 1.0)))
            raw_warmup = max(0, min(raw_warmup, num_warmup_steps))

            # invert: larger factor -> shorter warmup_layer, smaller factor -> longer warmup_layer
            warmup_layer = num_warmup_steps - raw_warmup
            warmup_layer = max(1, min(warmup_layer, num_training_steps))

            if current_step < warmup_layer:
                base = float(current_step) / float(max(1, warmup_layer))
                return factor * base
        else:
            warmup_layer = 0

        if current_step >= num_training_steps:
            return 0.0

        decay_den = max(1, num_training_steps - warmup_layer)
        progress = float(current_step - warmup_layer) / float(decay_den)
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (
            1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)
        )

        return factor * cosine

    lr_lambdas = []
    for group in optimizer.param_groups:
        llrd = float(group.get("llrd", 1.0))
        llrd_scale = float(group.get("llrd_scale", 0.0))
        lr_lambdas.append(lambda step, llrd=llrd, llrd_scale=llrd_scale: lr_lambda(step, llrd, llrd_scale))
    return LambdaLR(optimizer, lr_lambdas, last_epoch=last_epoch)
