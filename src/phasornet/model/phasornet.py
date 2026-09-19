

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class PhasorNetConfig:
    num_phasors: int = 3
    num_layers: int = 2
    observation_freqs: tuple[float, ...] = (1 / 32.0, 1 / 16.0, 0.0)
    observable_dims: tuple[int, ...] = (64, 128)
    state_dims: tuple[int, ...] = (128, 128)
    num_modes: tuple[int, ...] = (8, 8)
    hidden_dims: tuple[int, ...] = (256, 256)
    dt: float = 0.05

    def __post_init__(self):
        if len(self.observation_freqs) < self.num_phasors:
            raise ValueError(
                f"Length of observation_freqs ({len(self.observation_freqs)}) "
                f"must be >= num_phasors ({self.num_phasors})."
            )

        layer_configs = [
            ("observable_dims", self.observable_dims),
            ("state_dims", self.state_dims),
            ("num_modes", self.num_modes),
            ("hidden_dims", self.hidden_dims),
        ]
        for name, cfg_tuple in layer_configs:
            if len(cfg_tuple) < self.num_layers:
                raise ValueError(
                    f"Length of {name} ({len(cfg_tuple)}) must be >= num_layers ({self.num_layers})."
                )

        for l in range(1, self.num_layers):
            if self.observable_dims[l] != self.state_dims[l - 1]:
                raise ValueError(
                    f"Layer {l} observable_dim ({self.observable_dims[l]}) must match "
                    f"layer {l-1} state_dim ({self.state_dims[l - 1]})."
                )


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
    return psi * torch.sqrt(target_norm / norm.clamp_min(eps))


class InteractionPotential(nn.Module):
    """Inter-mode potential within a single Phasor."""

    def forward(self, wavefunctions: torch.Tensor) -> torch.Tensor:
        total_conj = torch.conj(wavefunctions).sum(dim=1)
        return (wavefunctions * total_conj.unsqueeze(1)).real


class LocalPotential(nn.Module):
    """Local probability-density potential."""

    def forward(self, wavefunctions: torch.Tensor) -> torch.Tensor:
        return torch.abs(wavefunctions).pow(2).sum(dim=1)


class KineticOperator(nn.Module):
    """Learnable Hermitian kinetic operator."""

    def __init__(self, state_dim: int):
        super().__init__()

        self.real = nn.Parameter(torch.randn(state_dim, state_dim) * 0.02)
        self.imag = nn.Parameter(torch.randn(state_dim, state_dim) * 0.02)

    def forward(self) -> torch.Tensor:
        T = torch.complex(self.real, self.imag)
        return 0.5 * (T + T.mH)


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

        V = (
            self.local(wavefunctions).unsqueeze(1)
            + self.interaction(wavefunctions)
        )

        if external_potential is not None:
            V = V + external_potential.unsqueeze(1)

        H_psi = torch.matmul(wavefunctions, T.T) + V * wavefunctions

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
    """Learned complex normal modes."""

    def __init__(self, state_dim: int, num_modes: int):
        super().__init__()

        self.real = nn.Parameter(torch.randn(state_dim, num_modes) * 0.02)
        self.imag = nn.Parameter(torch.randn(state_dim, num_modes) * 0.02)

    def forward(self) -> torch.Tensor:
        return torch.complex(self.real, self.imag)


class ModeExcitations(nn.Module):
    """Maps observations to complex mode weights."""

    def __init__(self, observable_dim: int, hidden_dim: int, num_modes: int):
        super().__init__()

        self.mode_weights = nn.Sequential(
            nn.Linear(observable_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_modes * 2),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        real, imag = self.mode_weights(observation).chunk(2, dim=-1)
        return torch.complex(real, imag)


class StateSuperposition(nn.Module):
    """Combines latent state with observation."""

    def __init__(
        self,
        state_dim: int,
        observable_dim: int,
        hidden_dim: int,
        num_modes: int,
    ):
        super().__init__()

        self.num_modes = num_modes
        self.normal_modes = NormalModes(state_dim, num_modes)
        self.mode_excitations = ModeExcitations(
            observable_dim, hidden_dim, num_modes
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

        psi_observed = excitations.unsqueeze(-1) * modes.T.unsqueeze(0)
        psi = self.alpha * psi_latent + self.beta * psi_observed

        return normalize_state(psi, float(self.num_modes))


class Propagator(nn.Module):
    """Evolves latent modes through the Hamiltonian."""

    def __init__(self, state_dim: int, num_modes: int):
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

        psi_next = torch.exp(-1j * energy * dt).unsqueeze(-1) * psi

        return normalize_state(psi_next, float(self.num_modes))

class Phasor(nn.Module):

    def __init__(
        self,
        state_dim: int,
        observable_dim: int,
        hidden_dim: int,
        num_modes: int,
        dt: float,
        observation_freq: float,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_modes = num_modes
        self.dt = dt
        self.observation_freq = observation_freq
        self.is_periodic = observation_freq > 0.0

        if self.is_periodic:
            self.observation_interval = 1.0 / observation_freq
        else:
            self.observation_interval = None

        # Buffer retained for state_dict consistency across all Phasor instances
        self.register_buffer("time", torch.tensor(0.0), persistent=True)
        self.register_buffer("psi_latent", None, persistent=False)

        self.superposition = StateSuperposition(
            state_dim=state_dim,
            observable_dim=observable_dim,
            hidden_dim=hidden_dim,
            num_modes=num_modes,
        )

        self.propagator = Propagator(
            state_dim=state_dim,
            num_modes=num_modes,
        )

    def init_state(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.complex64,
    ):
        self.psi_latent = torch.zeros(
            (batch_size, self.num_modes, self.state_dim),
            dtype=dtype,
            device=device,
        )

    def reset_state(
        self,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
    ):
        self.time.zero_()
        if batch_size is not None:
            dev = device if device is not None else self.time.device
            self.init_state(batch_size, device=dev)
        else:
            self.psi_latent = None

    def detach_state(self):
        if self.psi_latent is not None:
            self.psi_latent = self.psi_latent.detach()

    def collapse(self, modes: torch.Tensor) -> torch.Tensor:
        return torch.abs(modes.sum(dim=1)).pow(2)

    def forward(
        self,
        observation: torch.Tensor,
        external_potential: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = observation.shape[0]
        if self.psi_latent is None or self.psi_latent.shape[0] != batch_size:
            self.init_state(
                batch_size=batch_size,
                device=observation.device,
            )

        # Apply periodic modulation only if frequency > 0
        if self.is_periodic:
            obs_scaled = observation * torch.cos(
                2.0 * torch.pi * self.observation_freq * self.time
            ).pow(2)
        else:
            obs_scaled = observation

        psi = self.superposition(
            obs_scaled,
            self.psi_latent,
        )

        modes = self.propagator(
            psi,
            external_potential,
            self.dt,
        )

        # Update persistent latent state
        self.psi_latent = modes

        # Advance and wrap time buffer only if periodic
        if self.is_periodic:
            with torch.no_grad():
                self.time.add_(self.dt).remainder_(self.observation_interval)

        return modes, self.collapse(modes)


class PhasorStack(nn.Module):
    """P0 -> C0 -> P1 -> C1 -> ... -> Pn -> Cn."""

    def __init__(self, config: PhasorNetConfig, layer: int):
        super().__init__()

        self.phasors = nn.ModuleList([
            Phasor(
                state_dim=config.state_dims[layer],
                observable_dim=config.observable_dims[layer],
                hidden_dim=config.hidden_dims[layer],
                num_modes=config.num_modes[layer],
                dt=config.dt,
                observation_freq=config.observation_freqs[phasor_idx],
            )
            for phasor_idx in range(config.num_phasors)
        ])

    def forward(
        self,
        observation: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:

        results = []
        external_potential = None

        for phasor in self.phasors:
            modes, collapsed = phasor(
                observation=observation,
                external_potential=external_potential,
            )

            results.append((modes, collapsed))
            external_potential = collapsed

        return results


class PhasorNet(nn.Module):
    """Sequential hierarchy of PhasorStacks with persistent internal states."""

    def __init__(self, config: PhasorNetConfig):
        super().__init__()

        self.config = config
        self.stacks = nn.ModuleList([
            PhasorStack(config=config, layer=layer)
            for layer in range(config.num_layers)
        ])
        self.observable_projection = nn.Linear(
            config.state_dims[-1],
            config.observable_dims[0],
        )

    def reset_states(
        self,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
    ):
        """Resets time and latent states across all Phasors in the model."""
        for stack in self.stacks:
            for phasor in stack.phasors:
                phasor.reset_state(batch_size=batch_size, device=device)

    def detach_states(self):
        """Detaches latent states from autograd computational graphs (TBPTT)."""
        for stack in self.stacks:
            for phasor in stack.phasors:
                phasor.detach_state()

    def forward(
        self,
        observation: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        list[list[tuple[torch.Tensor, torch.Tensor]]],
    ]:
        stack_observation = observation
        all_results = []

        for stack in self.stacks:
            results = stack(observation=stack_observation)
            all_results.append(results)
            stack_observation = results[-1][1]

        final_collapse = all_results[-1][-1][1]
        final_observable = self.observable_projection(final_collapse)

        return final_observable, all_results
