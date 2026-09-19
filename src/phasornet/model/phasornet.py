from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PhasorNetConfig:
    num_phasors: int = 3
    observation_intervals: tuple[int, ...] = (32, 16, 1)

    observation_dims: tuple[int, ...] = (64, 64, 64)
    state_dims: tuple[int, ...] = (128, 128, 128)
    num_modes: tuple[int, ...] = (16, 16, 16)
    hidden_dims: tuple[int, ...] = (256, 256, 256)

    num_layers: int = 2
    dt: float = 0.05


def normalize_state(
    psi: torch.Tensor,
    target_norm: float,
) -> torch.Tensor:
    norm = torch.sum(
        torch.abs(psi) ** 2,
        dim=(-2, -1),
        keepdim=True,
    )
    eps = torch.finfo(psi.real.dtype).eps
    scale = torch.sqrt(target_norm / norm.clamp_min(eps))
    return psi * scale


class InteractionPotential(nn.Module):
    """V_i = sum_j 0.5 * (psi_j* psi_i + psi_i* psi_j)."""

    def forward(self, wavefunctions: torch.Tensor) -> torch.Tensor:
        total_conj = torch.conj(wavefunctions).sum(dim=1)
        return (wavefunctions * total_conj.unsqueeze(1)).real


class LocalPotential(nn.Module):
    """V_local = sum_j |psi_j|^2."""

    def forward(self, wavefunctions: torch.Tensor) -> torch.Tensor:
        return torch.abs(wavefunctions).pow(2).sum(dim=1)


class KineticOperator(nn.Module):
    """Learnable Hermitian kinetic operator."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.real = nn.Parameter(
            torch.randn(state_dim, state_dim) * 0.02
        )
        self.imag = nn.Parameter(
            torch.randn(state_dim, state_dim) * 0.02
        )

    def forward(self) -> torch.Tensor:
        T = torch.complex(self.real, self.imag)
        return 0.5 * (T + torch.conj(T.T))


class Hamiltonian(nn.Module):
    """Computes real expectation energies."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.kinetic = KineticOperator(state_dim)
        self.local = LocalPotential()
        self.interaction = InteractionPotential()

    def forward(
        self,
        wavefunctions: torch.Tensor,
        external_potential: torch.Tensor | None = None,
    ) -> torch.Tensor:
        T = self.kinetic()

        V_local = self.local(wavefunctions)
        V_interaction = self.interaction(wavefunctions)

        V = V_local.unsqueeze(1) + V_interaction

        if external_potential is not None:
            V = V + external_potential.unsqueeze(1)

        T_psi = torch.matmul(wavefunctions, T.T)
        H_psi = T_psi + V * wavefunctions

        numerator = torch.sum(
            torch.conj(wavefunctions) * H_psi,
            dim=-1,
        )
        norm = torch.sum(
            torch.abs(wavefunctions) ** 2,
            dim=-1,
        )

        eps = torch.finfo(wavefunctions.real.dtype).eps
        return (numerator / norm.clamp_min(eps)).real


class NormalModes(nn.Module):
    """Learned persistent complex normal modes."""

    def __init__(self, state_dim: int, num_modes: int):
        super().__init__()
        self.real = nn.Parameter(
            torch.randn(state_dim, num_modes) * 0.02
        )
        self.imag = nn.Parameter(
            torch.randn(state_dim, num_modes) * 0.02
        )

    def forward(self) -> torch.Tensor:
        return torch.complex(self.real, self.imag)


class ModeExcitations(nn.Module):
    """Maps observations to complex mode weights."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int,
        num_modes: int,
    ):
        super().__init__()
        self.mode_weights = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_modes * 2),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        real, imag = self.mode_weights(observation).chunk(2, dim=-1)
        return torch.complex(real.float(), imag.float())


class StateSuperposition(nn.Module):
    """Excites and normalizes the latent modes."""

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
            observation_dim,
            hidden_dim,
            num_modes,
        )

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        observation: torch.Tensor,
        psi_latent: torch.Tensor,
    ) -> torch.Tensor:
        modes = self.normal_modes()
        excitations = self.mode_excitations(observation)

        psi_observed = (
            excitations.unsqueeze(-1)
            * modes.T.unsqueeze(0)
        )

        psi = self.alpha * psi_latent + self.beta * psi_observed

        return normalize_state(psi, self.num_modes)


class Propagator(nn.Module):
    """Evolves latent modes through the Hamiltonian."""

    def __init__(
        self,
        state_dim: int,
        num_modes: int,
    ):
        super().__init__()

        self.num_modes = num_modes
        self.hamiltonian = Hamiltonian(state_dim)

    def forward(
        self,
        psi: torch.Tensor,
        external_potential: torch.Tensor | None,
        dt: float,
    ) -> torch.Tensor:
        energy = self.hamiltonian(psi, external_potential)
        evolution = torch.exp(1j * energy * dt)
        psi_next = evolution.unsqueeze(-1) * psi

        return normalize_state(psi_next, self.num_modes)


class Phasor(nn.Module):
    """
    One complete dynamical unit.

    Returns:
        modes: [B, N, D]
        collapse: [B, D]
    """

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
            state_dim=state_dim,
            observation_dim=observation_dim,
            hidden_dim=hidden_dim,
            num_modes=num_modes,
        )

        self.propagator = Propagator(
            state_dim=state_dim,
            num_modes=num_modes,
        )

    def collapse(self, modes: torch.Tensor) -> torch.Tensor:
        return torch.abs(modes.sum(dim=1)).pow(2)

    def forward(
        self,
        observation: torch.Tensor,
        psi_latent: torch.Tensor,
        external_potential: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        psi = self.superposition(
            observation,
            psi_latent,
        )

        modes = self.propagator(
            psi,
            external_potential,
            self.dt,
        )

        return modes, self.collapse(modes)


class PhasorStack(nn.Module):
    """
    One hierarchical layer of Phasors.

    P0 -> C0 -> P1 -> C1 -> ... -> Pn -> Cn
    """

    def __init__(
        self,
        config: PhasorNetConfig,
        input_dim: int,
    ):
        super().__init__()

        self.observation_intervals = config.observation_intervals

        self.input_projections = nn.ModuleList([
            nn.Linear(
                input_dim,
                config.observation_dims[i],
            )
            for i in range(config.num_phasors)
        ])

        self.phasors = nn.ModuleList([
            Phasor(
                state_dim=config.state_dims[i],
                observation_dim=config.observation_dims[i],
                hidden_dim=config.hidden_dims[i],
                num_modes=config.num_modes[i],
                dt=config.dt,
            )
            for i in range(config.num_phasors)
        ])

    def forward(
        self,
        observation: torch.Tensor,
        psi_latents: list[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        results = []
        external_potential = None

        for i, phasor in enumerate(self.phasors):
            phasor_observation = self.input_projections[i](observation)

            modes, collapsed = phasor(
                observation=phasor_observation,
                psi_latent=psi_latents[i],
                external_potential=external_potential,
            )

            results.append((modes, collapsed))
            external_potential = collapsed

        return results


class PhasorNet(nn.Module):
    """
    Sequential hierarchy of PhasorStacks.

    Observation
        -> Stack 0
        -> Stack 1
        -> ...
        -> final fast collapse
    """

    def __init__(self, config: PhasorNetConfig):
        super().__init__()

        self.config = config

        self.stacks = nn.ModuleList()

        input_dim = config.observation_dims[0]

        for layer in range(config.num_layers):
            stack = PhasorStack(
                config=config,
                input_dim=input_dim,
            )
            self.stacks.append(stack)

            input_dim = config.state_dims[-1]

    def forward(
        self,
        observation: torch.Tensor,
        psi_latents: list[list[torch.Tensor]],
    ) -> tuple[
        torch.Tensor,
        list[list[tuple[torch.Tensor, torch.Tensor]]],
    ]:
        stack_input = observation
        all_results = []

        for layer, stack in enumerate(self.stacks):
            results = stack(
                observation=stack_input,
                psi_latents=psi_latents[layer],
            )

            all_results.append(results)

            # Only the final Phasor collapse is forwarded
            # to the next PhasorStack.
            stack_input = results[-1][1]

        final_state = all_results[-1][-1][1]

        return final_state, all_results
