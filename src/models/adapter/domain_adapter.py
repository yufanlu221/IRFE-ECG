"""Domain-isolated adapter model for continual ECG classification.

The backbone is frozen. Each domain owns both a bottleneck adapter and a
fast-slow classifier head. Keeping heads per domain is important: a shared head
can overwrite old-domain decision boundaries even when adapters are isolated.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from models.head.fast_slow_head import FastSlowHead


class DomainAdapter(nn.Module):
    """Bottleneck residual adapter: LN -> down -> GELU -> up -> residual."""

    def __init__(self, embed_dim: int = 160, bottleneck: int = 32) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, embed_dim)

        # Start as an identity residual branch.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(self.norm(x))))


class AdapterCLModel(nn.Module):
    """Frozen backbone + per-domain adapter + per-domain fast-slow head."""

    def __init__(
        self,
        backbone: nn.Module,
        domain_names: Optional[List[str]] = None,
        embed_dim: int = 160,
        bottleneck: int = 32,
        num_classes: int = 2,
        ema_beta: float = 0.9,
        fast_alpha: float = 0.3,
        hook_stage: int = 2,
        num_tasks: Optional[int] = None,
    ) -> None:
        super().__init__()
        if domain_names is None:
            if num_tasks is None:
                raise ValueError("Either domain_names or num_tasks must be provided.")
            domain_names = [f"task_{idx}" for idx in range(num_tasks)]
        if not domain_names:
            raise ValueError("domain_names must not be empty.")

        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.domain_names = list(domain_names)
        self.domain_to_index = {name: idx for idx, name in enumerate(self.domain_names)}
        self.embed_dim = int(embed_dim)
        self.hook_stage = int(hook_stage)
        self.current_domain = self.domain_names[0]
        self._hook_feature: Optional[torch.Tensor] = None

        self.adapters = nn.ModuleDict(
            {name: DomainAdapter(embed_dim, bottleneck) for name in self.domain_names}
        )
        self.heads = nn.ModuleDict(
            {
                name: FastSlowHead(embed_dim, num_classes, ema_beta, fast_alpha)
                for name in self.domain_names
            }
        )

        if not hasattr(self.backbone, "stage_list"):
            raise AttributeError("backbone must expose stage_list for feature hooking.")
        if self.hook_stage >= len(self.backbone.stage_list):
            raise ValueError(
                f"hook_stage={self.hook_stage} out of range for "
                f"{len(self.backbone.stage_list)} stages."
            )
        self._hook = self.backbone.stage_list[self.hook_stage].register_forward_hook(
            self._capture_hook
        )

        self.set_domain(self.current_domain, verbose=False)

    @property
    def head(self) -> FastSlowHead:
        """Current domain head, kept for backward-compatible trainer code."""
        return self.heads[self.current_domain]

    def _capture_hook(self, module: nn.Module, inp, out: torch.Tensor) -> None:
        if out.ndim != 3:
            raise RuntimeError(
                f"Expected hook output with shape (B, C, L), got {tuple(out.shape)}"
            )
        # Redundant with extract_features() no_grad, but documents the boundary:
        # gradients must stop at the frozen backbone and start at the adapter.
        self._hook_feature = out.mean(dim=-1).detach()

    def train(self, mode: bool = True) -> "AdapterCLModel":
        """Set training mode while keeping the frozen backbone in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def _resolve_domain(
        self,
        domain: Optional[str] = None,
        task_id: Optional[int] = None,
    ) -> str:
        if domain is not None and task_id is not None:
            raise ValueError("Pass either domain or task_id, not both.")
        if task_id is not None:
            try:
                domain = self.domain_names[int(task_id)]
            except IndexError as exc:
                raise ValueError(f"Unknown task_id={task_id}") from exc
        domain = domain or self.current_domain
        if domain not in self.adapters:
            raise ValueError(f"Unknown domain '{domain}'. Choices: {self.domain_names}")
        return domain

    def set_domain(self, domain: str, verbose: bool = True) -> None:
        """Activate one domain and freeze all other adapters/heads."""
        domain = self._resolve_domain(domain=domain)
        self.current_domain = domain

        for name, adapter in self.adapters.items():
            requires_grad = name == domain
            for p in adapter.parameters():
                p.requires_grad = requires_grad

        for name, head in self.heads.items():
            requires_grad = name == domain
            for p in head.parameters():
                p.requires_grad = requires_grad

        if verbose:
            print(f"  [Domain] active={domain}; adapters/heads for other domains frozen")

    def set_task(self, task_id: int) -> None:
        """Backward-compatible alias for older task-id based scripts."""
        self.set_domain(self._resolve_domain(task_id=task_id))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run frozen backbone and return GAP features from the hook stage."""
        self._hook_feature = None
        with torch.no_grad():
            _ = self.backbone(x)
        if self._hook_feature is None:
            raise RuntimeError("Feature hook did not fire. Check hook_stage/backbone.")
        if self._hook_feature.shape[1] != self.embed_dim:
            raise RuntimeError(
                f"Hook feature dim mismatch: expected {self.embed_dim}, "
                f"got {self._hook_feature.shape[1]}."
            )
        return self._hook_feature

    def forward(
        self,
        x: torch.Tensor,
        domain: Optional[str] = None,
        task_id: Optional[int] = None,
        use_merge: bool = False,
    ) -> torch.Tensor:
        domain = self._resolve_domain(domain=domain, task_id=task_id)
        if self.training and domain != self.current_domain:
            raise RuntimeError(
                "Training forward must use current_domain. Call set_domain(domain) "
                "before rebuilding the optimizer, or switch to eval() for "
                "cross-domain evaluation."
            )
        feat = self.extract_features(x)
        return self.forward_features(feat, domain=domain, use_merge=use_merge)

    def forward_features(
        self,
        feat: torch.Tensor,
        domain: Optional[str] = None,
        task_id: Optional[int] = None,
        use_merge: bool = False,
    ) -> torch.Tensor:
        """Classify pre-extracted frozen-backbone features for one domain."""
        domain = self._resolve_domain(domain=domain, task_id=task_id)
        if self.training and domain != self.current_domain:
            raise RuntimeError(
                "Training feature forward must use current_domain. Call "
                "set_domain(domain), or switch to eval() for cross-domain routing."
            )
        feat = self.adapters[domain](feat)
        return self.heads[domain](feat, use_merge=use_merge)

    def update_slow(self, domain: Optional[str] = None) -> None:
        """Update EMA buffers for one domain head."""
        domain = self._resolve_domain(domain=domain)
        self.heads[domain].update_slow()

    def reset_slow(self, domain: Optional[str] = None) -> None:
        """Reset EMA buffers for one domain head."""
        domain = self._resolve_domain(domain=domain)
        self.heads[domain].reset_slow()

    def set_head_hparams(
        self,
        domain: Optional[str] = None,
        ema_beta: Optional[float] = None,
        fast_alpha: Optional[float] = None,
    ) -> None:
        """Update fast-slow head hyperparameters for one domain."""
        domain = self._resolve_domain(domain=domain)
        head = self.heads[domain]
        if ema_beta is not None:
            if not 0.0 <= ema_beta < 1.0:
                raise ValueError(f"ema_beta must be in [0, 1), got {ema_beta}")
            head.ema_beta = float(ema_beta)
        if fast_alpha is not None:
            if not 0.0 <= fast_alpha <= 1.0:
                raise ValueError(f"fast_alpha must be in [0, 1], got {fast_alpha}")
            head.fast_alpha = float(fast_alpha)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        params.extend(self.adapters[self.current_domain].parameters())
        params.extend(self.heads[self.current_domain].parameters())
        return [p for p in params if p.requires_grad]

    def get_task_params(self) -> List[nn.Parameter]:
        """Backward-compatible alias for older task-id based scripts."""
        return self.get_trainable_params()

    def active_parameter_count(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def per_domain_parameter_count(self, domain: Optional[str] = None) -> int:
        domain = self._resolve_domain(domain=domain)
        return sum(p.numel() for p in self.adapters[domain].parameters()) + sum(
            p.numel() for p in self.heads[domain].parameters()
        )

    def remove_hook(self) -> None:
        self._hook.remove()

    def to_device(self, device: torch.device) -> "AdapterCLModel":
        return self.to(device)
