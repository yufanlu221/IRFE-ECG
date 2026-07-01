"""Fast-slow linear classification head.

The fast stream is a trainable linear classifier. The slow stream is an EMA
copy of the fast weights and is stored as buffers, so it is checkpointed but
not optimized directly.

In continual learning experiments this module should usually be instantiated
per domain/task. Sharing one instance across domains mixes the EMA states and
weakens the parameter-isolation claim.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSlowHead(nn.Module):
    """Linear head with a trainable fast stream and an EMA slow stream."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        ema_beta: float = 0.9,
        fast_alpha: float = 0.3,
    ) -> None:
        super().__init__()
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError(f"ema_beta must be in [0, 1), got {ema_beta}")
        if not 0.0 <= fast_alpha <= 1.0:
            raise ValueError(f"fast_alpha must be in [0, 1], got {fast_alpha}")

        self.ema_beta = float(ema_beta)
        self.fast_alpha = float(fast_alpha)

        self.W_fast = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.W_fast.weight)
        nn.init.zeros_(self.W_fast.bias)

        self.register_buffer("W_slow_weight", self.W_fast.weight.detach().clone())
        self.register_buffer("W_slow_bias", self.W_fast.bias.detach().clone())
        self.register_buffer("_slow_initialized", torch.tensor(False))

    @property
    def slow_initialized(self) -> bool:
        return bool(self._slow_initialized.item())

    @torch.no_grad()
    def update_slow(self) -> None:
        """Update EMA buffers from the current fast weights."""
        w_fast = self.W_fast.weight.detach()
        b_fast = self.W_fast.bias.detach()

        if not self.slow_initialized:
            self.W_slow_weight.copy_(w_fast)
            self.W_slow_bias.copy_(b_fast)
            self._slow_initialized.fill_(True)
            return

        beta = self.ema_beta
        self.W_slow_weight.mul_(beta).add_(w_fast, alpha=1.0 - beta)
        self.W_slow_bias.mul_(beta).add_(b_fast, alpha=1.0 - beta)

    @torch.no_grad()
    def reset_slow(self) -> None:
        """Reset slow stream to the current fast weights.

        After reset the slow stream is immediately usable for merged inference.
        This avoids the surprising behavior where ``reset_slow(); forward(...,
        use_merge=True)`` silently ignores the slow stream.
        """
        self.W_slow_weight.copy_(self.W_fast.weight.detach())
        self.W_slow_bias.copy_(self.W_fast.bias.detach())
        self._slow_initialized.fill_(True)

    @torch.no_grad()
    def reinitialize_slow(self) -> None:
        """Mark slow stream uninitialized so the next update copies fast weights."""
        self._slow_initialized.fill_(False)

    def merged_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return detached fast-slow merged weights for inference."""
        if not self.slow_initialized:
            warnings.warn(
                "FastSlowHead slow stream is not initialized; using fast weights only.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.W_fast.weight.detach(), self.W_fast.bias.detach()

        alpha = self.fast_alpha
        weight = (
            alpha * self.W_fast.weight.detach()
            + (1.0 - alpha) * self.W_slow_weight
        )
        bias = alpha * self.W_fast.bias.detach() + (1.0 - alpha) * self.W_slow_bias
        return weight, bias

    def forward(self, x: torch.Tensor, use_merge: bool = False) -> torch.Tensor:
        """Compute logits.

        Args:
            x: Feature tensor with shape ``(batch, embed_dim)``.
            use_merge: If true, use detached fast-slow merged weights. This is
                intended for evaluation/inference.
        """
        if use_merge and self.training:
            raise RuntimeError(
                "use_merge=True is for eval/inference only; call model.eval() first."
            )
        if not use_merge:
            return self.W_fast(x)

        weight, bias = self.merged_parameters()
        return F.linear(x, weight, bias)
