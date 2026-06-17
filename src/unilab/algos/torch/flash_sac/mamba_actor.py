"""Mamba Actor for FlashSAC — selective state-space policy backbone.

Pure PyTorch implementation of Mamba-1 (selective SSM + 1D convolution).
When mamba-ssm is available, uses the official CUDA-accelerated implementation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from unilab.algos.torch.flash_sac.layers import NormalTanhPolicy

# ── try official mamba-ssm ──────────────────────────────────────────
try:
    from mamba_ssm import Mamba as _OfficialMamba

    _HAS_OFFICIAL_MAMBA = True
except ImportError:
    _HAS_OFFICIAL_MAMBA = False


# ═══════════════════════════════════════════════════════════════════════
# Pure PyTorch Selective SSM (Mamba-1)
# ═══════════════════════════════════════════════════════════════════════


class SelectiveSSM(nn.Module):
    """Selective State Space Model (S6) — the core of Mamba."""

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int | str = "auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        dt_rank_val = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        # Δ projection (input-dependent step size)
        self.dt_proj = nn.Linear(dt_rank_val, d_model)
        # Input-dependent Δ logit
        self.x_proj_dt = nn.Linear(d_model, dt_rank_val, bias=False)

        # A (state transition) — learned but not input-dependent
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, d_state)
        A = A.expand(d_model, -1)  # (d_model, d_state)
        self.A_log = nn.Parameter(torch.log(A))

        # B, C projections (input-dependent)
        self.x_proj_B = nn.Linear(d_model, d_state, bias=False)
        self.x_proj_C = nn.Linear(d_model, d_state, bias=False)

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        B, L, D = x.shape

        # Compute input-dependent parameters
        dt = F.softplus(self.dt_proj(self.x_proj_dt(x)))  # (B, L, D)
        A = -torch.exp(self.A_log).to(x.dtype)  # (D, d_state), negative for stability
        B_proj = self.x_proj_B(x)  # (B, L, d_state)
        C_proj = self.x_proj_C(x)  # (B, L, d_state)

        # Discretize: Ā = exp(Δ·A), B̄ = (Δ·A)⁻¹(exp(Δ·A)-I)·ΔB ≈ Δ·B
        # Simplified discretization (zero-order hold)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, d_state)
        dB = dt.unsqueeze(-1) * B_proj.unsqueeze(2)  # (B, L, D, d_state)

        # Parallel associative scan implementation
        y = self._selective_scan(x, dA, dB, C_proj)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y

    def _selective_scan(self, u, dA, dB, C):
        """Selective scan: parallel prefix sum over SSM states."""
        B_s, L_s, D_s, N_s = dA.shape
        h = torch.zeros(B_s, D_s, N_s, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L_s):
            h = dA[:, i] * h + dB[:, i] * u[:, i].unsqueeze(-1)
            ys.append((h * C[:, i].unsqueeze(1)).sum(dim=-1))  # (B, D)
        return torch.stack(ys, dim=1)  # (B, L, D)


# ═══════════════════════════════════════════════════════════════════════
# Mamba Block
# ═══════════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MambaBlock(nn.Module):
    """Single Mamba block: RMSNorm → SSM → residual."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_official: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        inner_dim = d_model * expand
        self.use_official = use_official and _HAS_OFFICIAL_MAMBA

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner_dim * 2, bias=False)  # x + z branches

        if self.use_official:
            self.mamba = _OfficialMamba(
                d_model=inner_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=1,  # already expanded by in_proj
            )
        else:
            self.conv1d = nn.Conv1d(inner_dim, inner_dim, d_conv, padding=d_conv - 1, groups=inner_dim)
            self.ssm = SelectiveSSM(inner_dim, d_state)
            self.activation = nn.SiLU()

        self.out_proj = nn.Linear(inner_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        residual = x
        x = self.norm(x)

        # Project
        xz = self.in_proj(x)  # (B, L, 2*inner_dim)
        x_in, z = xz.chunk(2, dim=-1)

        if self.use_official:
            x_out = self.mamba(x_in)
        else:
            # Conv1d (causal padding → trim)
            L = x_in.shape[1]
            x_conv = self.conv1d(x_in.transpose(1, 2))  # (B, inner_dim, L+pad-1)
            x_conv = x_conv[..., :L].transpose(1, 2)  # (B, L, inner_dim)
            x_conv = self.activation(x_conv)
            x_out = self.ssm(x_conv)

        # Gate with z (SiLU)
        x_out = x_out * F.silu(z)
        x_out = self.out_proj(x_out)
        return x_out + residual


# ═══════════════════════════════════════════════════════════════════════
# Mamba Actor
# ═══════════════════════════════════════════════════════════════════════


class MambaActor(nn.Module):
    """Mamba-based stochastic policy for continuous control.

    Architecture:
        obs → TokenEmbed → Mamba blocks → mean-pool → TanhNormal head

    The selective state space allows the policy to naturally switch
    between different behavioral modes (walk/stand/flamingo/push-resist)
    without explicit skill conditioning.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 256,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_tokens: int = 4,
        noise_zeta_mu: float = 2.0,
        noise_zeta_max: int = 16,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.n_tokens = n_tokens

        # Token embedding: split obs into n_tokens tokens
        self.token_proj = nn.Linear(obs_dim, d_model * n_tokens)
        # Position encoding (learnable)
        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        # Mamba blocks
        self.blocks = nn.ModuleList(
            [
                MambaBlock(d_model, d_state, d_conv, expand)
                for _ in range(n_layers)
            ]
        )
        self.post_norm = RMSNorm(d_model)

        # Policy head
        self.predictor = NormalTanhPolicy(hidden_dim=d_model, action_dim=action_dim)

        # Exploration noise (same as FlashSACActor)
        self.noise_zeta_mu = noise_zeta_mu
        self.noise_zeta_max = noise_zeta_max
        ns = torch.arange(1, noise_zeta_max + 1, dtype=torch.float32)
        pmf = ns.pow(-noise_zeta_mu)
        self.register_buffer("zeta_cdf", torch.cumsum(pmf / pmf.sum(), dim=0))
        self.register_buffer("_noise", torch.zeros(0), persistent=False)
        self.register_buffer("_repeat_count", torch.zeros(0, dtype=torch.int32), persistent=False)
        self.register_buffer("_repeat_target", torch.zeros(0, dtype=torch.int32), persistent=False)

        self.to(device)

    def normalize_parameters(self) -> None:
        """No-op weight normalization (required by FlashSAC learner API)."""
        pass

    def as_export_module(self) -> "nn.Module":
        """Return a wrapper for ONNX export (required by FlashSAC)."""
        actor = self

        class _Wrapper(nn.Module):
            def forward(self, obs):
                mean, _ = actor.get_mean_and_std(obs, training=False)
                return torch.tanh(mean)

        return _Wrapper()

    def _encode(self, obs: torch.Tensor, training: bool) -> torch.Tensor:
        """Encode observation through Mamba backbone."""
        B = obs.shape[0]
        x = self.token_proj(obs)  # (B, d_model * n_tokens)
        x = x.reshape(B, self.n_tokens, self.d_model)  # (B, L, d_model)
        x = x + self.pos_emb

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim=1)  # (B, d_model)
        x = self.post_norm(x)
        return x

    def get_mean_and_std(
        self, obs: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self._encode(obs, training)
        return self.predictor.get_mean_and_std(encoded)

    def forward(
        self, obs: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = self._encode(obs, training)
        return self.predictor(encoded)

    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample actions matching FlashSACActor.explore() API."""
        if isinstance(dones, bool):
            deterministic = dones
            dones = None
        mean, std = self.get_mean_and_std(obs, training=False)
        if deterministic:
            return torch.tanh(mean)

        batch_size, action_dim = mean.shape
        self._ensure_exploration_state(batch_size, action_dim, mean.device, mean.dtype)

        if dones is None:
            done_mask = torch.zeros(batch_size, device=mean.device, dtype=torch.bool)
        else:
            done_mask = dones.to(device=mean.device).reshape(-1) > 0.5

        reinit = done_mask | (self._repeat_count <= 0) | (self._repeat_count >= self._repeat_target)
        if torch.any(reinit):
            new_noise = torch.randn_like(mean)
            new_target = self._sample_repeat_targets(batch_size, mean.device)
            self._noise = torch.where(reinit.unsqueeze(-1), new_noise, self._noise)
            self._repeat_target = torch.where(reinit, new_target, self._repeat_target)
            self._repeat_count = torch.where(
                reinit, torch.zeros_like(self._repeat_count), self._repeat_count
            )

        actions = torch.tanh(mean + std * self._noise)
        self._repeat_count += 1
        return actions

    def _ensure_exploration_state(self, batch_size, action_dim, device, dtype):
        if self._noise.numel() == 0 or self._noise.shape != (batch_size, action_dim):
            self._noise = torch.randn(batch_size, action_dim, device=device, dtype=dtype)
            self._repeat_count = torch.zeros(batch_size, device=device, dtype=torch.int32)
            self._repeat_target = self._sample_repeat_targets(batch_size, device)

    def _sample_repeat_targets(self, batch_size: int, device) -> torch.Tensor:
        idx = torch.searchsorted(self.zeta_cdf, torch.rand(batch_size, device=device))
        return idx.clamp(0, self.noise_zeta_max - 1).to(torch.int32)


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def create_mamba_actor(
    obs_dim: int = 98,
    action_dim: int = 29,
    d_model: int = 256,
    n_layers: int = 2,
    d_state: int = 16,
    n_tokens: int = 4,
    device: str = "cuda",
) -> MambaActor:
    """Factory to create a MambaActor with sensible defaults."""
    actor = MambaActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        n_tokens=n_tokens,
        device=device,
    )
    print(f"[MambaActor] params: {_count_params(actor):,}")
    print(f"[MambaActor] official mamba-ssm: {'available' if _HAS_OFFICIAL_MAMBA else 'not available (using pure PyTorch)'}")
    return actor
