import math

import torch

"""
Dictionary of supported optimization algorithms.
"""
OPTIMIZERS = {
    "adam": torch.optim.Adam,
}


class LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Learning-rate scheduler with warmup and decay-to-zero.

    Implementation derived from Megatron-LM's scheduler design.
    """

    DECAY_STYLES = {
        "constant": lambda x: 1.0,
        "linear": lambda x: 1.0 - x,
        "cosine": lambda x: 0.5 * (1 + math.cos(math.pi * x)),
    }

    def __init__(self, optimizer, start_lr, warmup_iter, num_iters, decay_style, args):
        self.optimizer = optimizer
        self.start_lr = float(start_lr)
        self.warmup_iter = warmup_iter
        self.num_iters = 0
        self.end_iter = num_iters
        self.decay_style = decay_style
        self.decay_func = LRScheduler.DECAY_STYLES[decay_style]
        self.args = args
        self.loss_annealing = "constant"
        self.step(self.num_iters)

    def get_lr(self):
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            return self.start_lr * self.num_iters / self.warmup_iter
        pct_step = (self.num_iters - self.warmup_iter) / (self.end_iter - self.warmup_iter)
        return self.start_lr * self.decay_func(pct_step)

    def get_loss_mult(self):
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            return 1.0 if self.loss_annealing == "constant" else 0.0

        adj_iter = self.num_iters - self.warmup_iter
        total_iter = self.end_iter - self.warmup_iter
        pct_step = adj_iter / total_iter
        if self.loss_annealing == "monotonic":
            return min(pct_step / self.loss_rate, 1.0)
        if self.loss_annealing == "cyclical":
            cycle = int(total_iter * self.loss_rate)
            return min(2.0 * ((adj_iter % cycle) / cycle), 1.0)
        return 1.0

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr * group.get("lr_scale", 1.0)

    def state_dict(self):
        return {
            "start_lr": self.start_lr,
            "warmup_iter": self.warmup_iter,
            "num_iters": self.num_iters,
            "decay_style": self.decay_style,
            "end_iter": self.end_iter,
        }

    def load_state_dict(self, sd):
        self.start_lr = sd["start_lr"]
        self.warmup_iter = sd["warmup_iter"]
        self.num_iters = sd["num_iters"]
        self.end_iter = sd["end_iter"]
        self.decay_style = sd["decay_style"]
        self.decay_func = LRScheduler.DECAY_STYLES[sd["decay_style"]]
        self.step(self.num_iters)


def get_lr_scheduler(optimizer, args, epoch_len):
    total_iterations = args.max_epoch * epoch_len
    warmup_iterations = math.floor(args.warmup_pct * total_iterations)

    return LRScheduler(
        optimizer=optimizer,
        start_lr=args.learning_rate,
        warmup_iter=warmup_iterations,
        num_iters=total_iterations,
        decay_style=args.lr_decay_style,
        args=args,
    )
