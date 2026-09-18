from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PhasorNetConfig:
    observation_dim: int
    state_dim: int = 256
    num_modes: int = 32
    hidden_dim: int = 512
    num_layers: int = 1
    dt: float = 0.05


def normalize_state(psi: torch.Tensor, target_norm: float) -> torch.Tensor:
    """Globally normalizes latent state to target_norm preserving relative mode amplitudes."""
    norm = torch.sum(torch.abs(psi) ** 2, dim=(-2, -1), keepdim=True)
    eps = torch.finfo(psi.real.dtype).eps
    scale = torch.sqrt(target_norm / norm.clamp_min(eps))
    return psi * scale


class InteractionPotential(nn.Module):
    """V_i = sum_j 0.5 * (psi_j* psi_i + psi_i* psi_j)."""

    def forward(self, wavefunctions: torch.Tensor) -> torch.Tensor:
        total_conj = torch.conj(wavefunctions).sum(dim=1)
        overlap = wavefunctions * total_conj.unsqueeze(1)
        return overlap.real


class LocalPotential(nn.Module):
    """V_local = sum_j |psi_j|^2."""
    def forward(self, wavefunctions: torch.Tensor) -> torch.Tensor:
        return torch.abs(wavefunctions).pow(2).sum(dim=1)


class KineticOperator(nn.Module):
    """Learnable Hermitian operator T = 0.5 * (T + T^dagger)."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.real = nn.Parameter(torch.randn(state_dim, state_dim) * 0.02)
        self.imag = nn.Parameter(torch.randn(state_dim, state_dim) * 0.02)

    def forward(self) -> torch.Tensor:
        T = torch.complex(self.real, self.imag)
        return 0.5 * (T + torch.conj(T.T))


class Hamiltonian(nn.Module):
    """Computes real expectation energies H = T + V."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.kinetic = KineticOperator(state_dim)
        self.local = LocalPotential()
        self.interaction = InteractionPotential()

    def forward(self, wavefunctions: torch.Tensor) -> torch.Tensor:
        # wavefunctions: [B, N, D]
        T = self.kinetic()                               # [D,D]
        V_local = self.local(wavefunctions)              # [B,D]
        V_interaction = self.interaction(wavefunctions)  # [B,N,D]

        V = V_local.unsqueeze(1) + V_interaction         # [B,N,D]

        T_psi = torch.matmul(wavefunctions, T.T)
        H_psi = T_psi + V * wavefunctions

        numerator = torch.sum(torch.conj(wavefunctions) * H_psi, dim=-1)
        norm = torch.sum(torch.abs(wavefunctions) ** 2, dim=-1)

        eps = torch.finfo(wavefunctions.real.dtype).eps
        energy = numerator / norm.clamp_min(eps)
        return energy.real


class NormalModes(nn.Module):
    """Learned persistent complex normal modes."""

    def __init__(self, state_dim: int, num_modes: int):
        super().__init__()
        self.real = nn.Parameter(torch.randn(state_dim, num_modes) * 0.02)
        self.imag = nn.Parameter(torch.randn(state_dim, num_modes) * 0.02)

    def forward(self) -> torch.Tensor:
        return torch.complex(self.real, self.imag)


class ModeExcitations(nn.Module):
    """Maps observation input to complex mode weights."""

    def __init__(self, observation_dim: int, hidden_dim: int, num_modes: int):
        super().__init__()
        self.mode_weights = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_modes * 2),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        weights = self.mode_weights(observation)
        real, imag = weights.chunk(2, dim=-1)

        real = real.float()
        imag = imag.float()
        return torch.complex(real, imag)


class StateSuperposition(nn.Module):
    """Excites modes via observation and enforces psi*psi = num_modes."""

    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        hidden_dim: int,
        num_modes: int,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.normal_modes = NormalModes(state_dim, num_modes)
        self.mode_excitations = ModeExcitations(
            observation_dim, hidden_dim, num_modes
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, observation: torch.Tensor, psi_latent: torch.Tensor
    ) -> torch.Tensor:
        modes = self.normal_modes()
        excitations = self.mode_excitations(observation)

        psi_observed = excitations.unsqueeze(-1) * modes.T.unsqueeze(0)
        psi_latent = self.alpha * psi_latent + self.beta * psi_observed

        return normalize_state(psi_latent, self.num_modes)


class Propagator(nn.Module):
    """Evolves latent state through expectation energies with norm constraint."""

    def __init__(self, config: PhasorNetConfig):
        super().__init__()
        self.num_modes = config.num_modes
        self.hamiltonian = Hamiltonian(config.state_dim)

    def forward(self, psi_latent: torch.Tensor, dt: float) -> torch.Tensor:
        energy = self.hamiltonian(psi_latent)
        evolution = torch.exp(1j * energy * dt)
        psi_next = evolution.unsqueeze(-1) * psi_latent

        return normalize_state(psi_next, self.num_modes)


class PhasorNetBase(nn.Module):
    """Single PhasorNet dynamical layer."""

    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        hidden_dim: int,
        num_modes: int,
        dt: float,
    ):
        super().__init__()
        self.dt = dt
        self.superposition = StateSuperposition(
            state_dim, observation_dim, hidden_dim, num_modes
        )
        self.propagator = Propagator(
            PhasorNetConfig(
                observation_dim=observation_dim,
                state_dim=state_dim,
                num_modes=num_modes,
                hidden_dim=hidden_dim,
                dt=dt,
            )
        )

    def forward(self, observation: torch.Tensor, psi_latent: torch.Tensor) -> torch.Tensor:
        psi_latent = self.superposition(observation, psi_latent)
        return self.propagator(psi_latent, self.dt)


class PhasorNet(nn.Module):
    """Stacked multi-layer PhasorNet architecture."""

    def __init__(self, config: PhasorNetConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()

        for layer_idx in range(config.num_layers):
            obs_dim = (
                config.observation_dim
                if layer_idx == 0
                else config.state_dim
            )
            self.layers.append(
                PhasorNetBase(
                    state_dim=config.state_dim,
                    observation_dim=obs_dim,
                    hidden_dim=config.hidden_dim,
                    num_modes=config.num_modes,
                    dt=config.dt,
                )
            )

        self.readout = nn.Linear(config.state_dim, config.observation_dim)

    def forward(self, observation: torch.Tensor, psi_latents: list):
        next_observation = observation
        next_psi_latents = []

        for layer, psi_latent in zip(self.layers, psi_latents):
            psi_latent = layer(next_observation, psi_latent)
            next_observation = torch.abs(psi_latent.sum(dim=1)) ** 2
            next_psi_latents.append(psi_latent)

        output_observation = self.readout(next_observation)
        return output_observation, next_psi_latents
