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


def normalize_state(psi, target_norm):
    """
    Globally normalize the complete latent state.

    psi = sum_i c_i * mode_i

    Enforces:
        psi* psi = target_norm

    One scalar is used for the entire state, so relative
    mode amplitudes are preserved.
    """
    norm = torch.sum(
        torch.abs(psi) ** 2,
        dim=(-2, -1),
        keepdim=True,
    )

    eps = torch.finfo(psi.real.dtype).eps

    scale = torch.sqrt(
        target_norm / norm.clamp_min(eps)
    )

    return psi * scale


class LocalPotential(nn.Module):
    """V(|psi_j|^2)."""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.real = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.imag = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, wavefunctions):
        x = torch.abs(wavefunctions) ** 2

        return torch.complex(
            self.real(x),
            self.imag(x),
        )


class InteractionPotential(nn.Module):
    """V(psi_j* ⊙ psi_i) ⊙ psi_j."""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.real = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.imag = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, wavefunctions):
        # [B, N, D]
        psi_i = wavefunctions.unsqueeze(2)
        psi_j = wavefunctions.unsqueeze(1)

        interaction = torch.conj(psi_j) * psi_i
        # [B, N, N, D]

        x = torch.cat(
            [interaction.real, interaction.imag],
            dim=-1,
        )

        potential = torch.complex(
            self.real(x),
            self.imag(x),
        )
        # [B, N, N, D]

        return potential * psi_j


class KineticOperator(nn.Module):
    """Learnable complex base operator T."""

    def __init__(self, state_dim):
        super().__init__()

        self.real = nn.Parameter(
            torch.randn(state_dim, state_dim) * 0.02
        )

        self.imag = nn.Parameter(
            torch.randn(state_dim, state_dim) * 0.02
        )

    def forward(self):
        return torch.complex(
            self.real,
            self.imag,
        )


class Hamiltonian(nn.Module):
    """Compute normalized complex expectation energies."""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.kinetic = KineticOperator(state_dim)

        self.local = LocalPotential(
            state_dim,
            hidden_dim,
        )

        self.interaction = InteractionPotential(
            state_dim,
            hidden_dim,
        )

    def forward(self, wavefunctions):
        # wavefunctions: [B, N, D]

        local = self.local(wavefunctions).sum(dim=1)
        # [B, D]

        kinetic = self.kinetic()
        # [D, D]

        kinetic_action = torch.matmul(
            wavefunctions,
            kinetic.T,
        )
        # [B, N, D]

        kinetic_energy = torch.sum(
            torch.conj(wavefunctions) * kinetic_action,
            dim=-1,
        )
        # [B, N]

        local_energy = torch.sum(
            torch.abs(wavefunctions) ** 2
            * local.unsqueeze(1),
            dim=-1,
        )
        # [B, N]

        exchange = self.interaction(
            wavefunctions
        ).sum(dim=2)
        # [B, N, D]

        exchange_energy = torch.sum(
            torch.conj(wavefunctions) * exchange,
            dim=-1,
        )
        # [B, N]

        numerator = (
            kinetic_energy
            + local_energy
            + exchange_energy
        )
        # [B, N]

        norm = torch.sum(
            torch.abs(wavefunctions) ** 2,
            dim=-1,
        )
        # [B, N]

        eps = torch.finfo(
            wavefunctions.real.dtype
        ).eps

        energy = numerator / norm.clamp_min(eps)
        # [B, N]

        return energy


class NormalModes(nn.Module):
    """Learned persistent complex normal modes."""

    def __init__(self, state_dim, num_modes):
        super().__init__()

        self.real = nn.Parameter(
            torch.randn(state_dim, num_modes) * 0.02
        )

        self.imag = nn.Parameter(
            torch.randn(state_dim, num_modes) * 0.02
        )

    def forward(self):
        return torch.complex(
            self.real,
            self.imag,
        )


class ModeExcitations(nn.Module):
    """Neural transformation: observation -> complex mode weights."""

    def __init__(self, observation_dim, hidden_dim, num_modes):
        super().__init__()

        self.mode_weights = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_modes * 2),
        )

    def forward(self, observation):
        weights = self.mode_weights(observation)

        real, imag = weights.chunk(2, dim=-1)

        return torch.complex(real, imag)


class StateSuperposition(nn.Module):
    """
    Observation excites learned modes.

    psi = sum_i c_i * mode_i

    The complete state is globally normalized so that:

        psi* psi = num_modes

    The same normalization factor is applied to every
    mode contribution.
    """

    def __init__(
        self,
        state_dim,
        observation_dim,
        hidden_dim,
        num_modes,
    ):
        super().__init__()

        self.num_modes = num_modes

        self.normal_modes = NormalModes(
            state_dim,
            num_modes,
        )

        self.mode_excitations = ModeExcitations(
            observation_dim,
            hidden_dim,
            num_modes,
        )

        self.alpha = nn.Parameter(
            torch.tensor(1.0)
        )

        self.beta = nn.Parameter(
            torch.tensor(1.0)
        )

    def forward(self, observation, psi_latent):
        modes = self.normal_modes()

        excitations = self.mode_excitations(
            observation
        )

        psi_observed = (
            excitations.unsqueeze(-1)
            * modes.T.unsqueeze(0)
        )
        # [B, N, D]

        psi_latent = (
            self.alpha * psi_latent
            + self.beta * psi_observed
        )

        psi_latent = normalize_state(
            psi_latent,
            self.num_modes,
        )

        return psi_latent


class Propagator(nn.Module):
    """
    Propagate all latent modes through their
    expectation energies.

    The propagated complete state is normalized
    to the fixed total norm afterward.
    """

    def __init__(self, config):
        super().__init__()

        self.num_modes = config.num_modes

        self.hamiltonian = Hamiltonian(
            config.state_dim,
            config.hidden_dim,
        )

    def forward(self, psi_latent, dt):
        # [B, N, D]

        energy = self.hamiltonian(
            psi_latent
        )
        # [B, N]

        evolution = torch.exp(
            1j * energy * dt
        )
        # [B, N]

        psi_next = (
            evolution.unsqueeze(-1)
            * psi_latent
        )
        # [B, N, D]

        psi_next = normalize_state(
            psi_next,
            self.num_modes,
        )

        return psi_next


class PhasorNetBase(nn.Module):
    """One independent PhasorNet dynamical layer."""

    def __init__(
        self,
        state_dim,
        observation_dim,
        hidden_dim,
        num_modes,
        dt,
    ):
        super().__init__()

        self.dt = dt

        self.superposition = StateSuperposition(
            state_dim=state_dim,
            observation_dim=observation_dim,
            hidden_dim=hidden_dim,
            num_modes=num_modes,
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

    def forward(self, observation, psi_latent):
        psi_latent = self.superposition(
            observation,
            psi_latent,
        )

        psi_latent = self.propagator(
            psi_latent,
            self.dt,
        )

        return psi_latent


class PhasorNet(nn.Module):
    """
    Stack of independent PhasorNet dynamical layers.

    Layer 0 receives the external observation.
    Each subsequent layer receives the collapsed
    state_dim representation from the previous layer:

        Phi_l = |sum_i psi_l,i|^2

    Every layer maintains:

        psi* psi = num_modes
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList()

        for layer_idx in range(config.num_layers):
            observation_dim = (
                config.observation_dim
                if layer_idx == 0
                else config.state_dim
            )

            self.layers.append(
                PhasorNetBase(
                    state_dim=config.state_dim,
                    observation_dim=observation_dim,
                    hidden_dim=config.hidden_dim,
                    num_modes=config.num_modes,
                    dt=config.dt,
                )
            )

        self.readout = nn.Linear(
            config.state_dim,
            config.observation_dim,
        )

    def forward(self, observation, psi_latents):
        next_observation = observation
        next_psi_latents = []

        for layer, psi_latent in zip(
            self.layers,
            psi_latents,
        ):
            psi_latent = layer(
                next_observation,
                psi_latent,
            )

            next_observation = (
                torch.abs(
                    psi_latent.sum(dim=1)
                ) ** 2
            )

            next_psi_latents.append(
                psi_latent
            )

        output_observation = self.readout(
            next_observation
        )

        return (
            output_observation,
            next_psi_latents,
        )
