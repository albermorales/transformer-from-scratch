"""Learning-rate schedule used by the original Transformer architecture."""

from __future__ import annotations

from typing import Any

from torch.optim import Optimizer


def transformer_learning_rate(
    step_num: float,
    d_model: int,
    warmup_steps: int,
) -> float:
    """Calculate the Transformer learning rate for a given training step.

    The implemented expression is::

        d_model**(-0.5) * min(
            step_num**(-0.5),
            step_num * warmup_steps**(-1.5),
        )

    ``step_num`` starts at 1 because the expression is undefined at zero.
    """
    if step_num < 1:
        raise ValueError("step_num must be at least 1")
    if d_model < 1:
        raise ValueError("d_model must be at least 1")
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be at least 1")

    return d_model ** -0.5 * min(
        step_num ** -0.5,
        step_num * warmup_steps ** -1.5,
    )


class TransformerLRScheduler:
    """Apply the Transformer schedule to every optimizer parameter group.

    The scheduler initializes the optimizer at step 1. Call ``step`` once
    after each ``optimizer.step()``. ``slowdown_factor`` stretches the curve
    horizontally without changing its initial or peak learning rates.
    Parameter groups can optionally define an ``lr_scale`` value to use
    different multiplicative learning rates.

    Example::

        optimizer = AdamW(model.parameters())
        scheduler = TransformerLRScheduler(
            optimizer,
            d_model=model.model_dim,
            warmup_steps=4_000,
        )

        for batch in train_dataloader:
            optimizer.zero_grad()
            loss = compute_batch(model, batch, criterion)
            loss.backward()
            optimizer.step()
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4_000,
        slowdown_factor: float = 1.0,
    ) -> None:
        if d_model < 1:
            raise ValueError("d_model must be at least 1")
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be at least 1")
        if slowdown_factor < 1:
            raise ValueError("slowdown_factor must be at least 1")

        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.slowdown_factor = slowdown_factor
        self.step_num = 1
        self._last_lr: list[float] = []
        self._apply_learning_rate()

    def step(self) -> float:
        """Advance one training step and update the optimizer learning rate."""
        self.step_num += 1
        self._apply_learning_rate()
        return self._last_lr[0]

    def get_last_lr(self) -> list[float]:
        """Return the learning rate currently used by each parameter group."""
        return self._last_lr.copy()

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state suitable for inclusion in a checkpoint."""
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "slowdown_factor": self.slowdown_factor,
            "step_num": self.step_num,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler state and update the optimizer immediately."""
        self.d_model = int(state_dict["d_model"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.slowdown_factor = float(state_dict.get("slowdown_factor", 1.0))
        self.step_num = int(state_dict["step_num"])

        if self.d_model < 1 or self.warmup_steps < 1 or self.slowdown_factor < 1 or self.step_num < 1:
            raise ValueError("invalid Transformer scheduler state")
        self._apply_learning_rate()

    def _apply_learning_rate(self) -> None:
        schedule_step = 1 + (self.step_num - 1) / self.slowdown_factor
        base_rate = transformer_learning_rate(
            schedule_step,
            self.d_model,
            self.warmup_steps,
        )
        self._last_lr = []
        for parameter_group in self.optimizer.param_groups:
            learning_rate = base_rate * float(parameter_group.get("lr_scale", 1.0))
            parameter_group["lr"] = learning_rate
            self._last_lr.append(learning_rate)
