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


class PhasorState:
    """External state container for PhasorNet latents and time counters."""

    def __init__(
        self,
        latents: list[list[torch.Tensor]],
        times: list[list[torch.Tensor]],
    ):
        self.latents = latents  # [num_layers][num_phasors] tensor
        self.times = times      # [num_layers][num_phasors] tensor

    @classmethod
    def create_zeros(
        cls,
        config: PhasorNetConfig,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.complex64,
    ) -> "PhasorState":
        """Factory method to initialize zero-states for a given batch size."""
        latents = [
            [
                torch.zeros(
                    (batch_size, config.num_modes[l], config.state_dims[l]),
                    dtype=dtype,
                    device=device,
                )
                for _ in range(config.num_phasors)
            ]
            for l in range(config.num_layers)
        ]
        times = [
            [
                torch.zeros((), dtype=torch.float32, device=device)
                for _ in range(config.num_phasors)
            ]
            for l in range(config.num_layers)
        ]
        return cls(latents=latents, times=times)

    def detach(self) -> "PhasorState":
        """Returns a new PhasorState detached from autograd computation graphs."""
        new_latents = [
            [psi.detach() for psi in layer_psi] for layer_psi in self.latents
        ]
        new_times = [
            [t.detach() for t in layer_t] for layer_t in self.times
        ]
        return PhasorState(latents=new_latents, times=new_times)

    def to(self, device: torch.device | str) -> "PhasorState":
        """Moves all state tensors to specified target device."""
        self.latents = [
            [psi.to(device) for psi in layer_psi] for layer_psi in self.latents
        ]
        self.times = [
            [t.to(device) for t in layer_t] for layer_t in self.times
        ]
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serializes internal state tensors to a flat dictionary."""
        sd = {}
        for l, layer in enumerate(self.latents):
            for p, psi in enumerate(layer):
                sd[f"layer_{l}.phasor_{p}.psi_latent"] = psi
                sd[f"layer_{l}.phasor_{p}.time"] = self.times[l][p]
        return sd

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        """Loads state tensors from a dictionary."""
        for l, layer in enumerate(self.latents):
            for p in range(len(layer)):
                self.latents[l][p] = state_dict[f"layer_{l}.phasor_{p}.psi_latent"]
                self.times[l][p] = state_dict[f"layer_{l}.phasor_{p}.time"]


def normalize_state(psi: torch.Tensor, target_norm: float) -> torch.Tensor:
    norm = torch.sum(
        torch.abs(psi) ** 2,
        dim=(-2, -1),
        keepdim=True,
    )
    eps = torch.finfo(psi.real.dtype).eps
    return psi * torch.sqrt(target_norm / norm.clamp_min(eps))


class InteractionPotential(nn.Module):
    def forward(self, wavefunctions: torch.Tensor) -> torch.Tensor:
        total_conj = torch.conj(wavefunctions).sum(dim=1)
        return (wavefunctions * total_conj.unsqueeze(1)).real


class LocalPotential(nn.Module):
    def forward(self, wavefunctions: torch.Tensor) -> torch.Tensor:
        return torch.abs(wavefunctions).pow(2).sum(dim=1)


class KineticOperator(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.real = nn.Parameter(torch.randn(state_dim, state_dim) * 0.02)
        self.imag = nn.Parameter(torch.randn(state_dim, state_dim) * 0.02)

    def forward(self) -> torch.Tensor:
        T = torch.complex(self.real, self.imag)
        return 0.5 * (T + T.mH)


class Hamiltonian(nn.Module):
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
        numerator = torch.sum(torch.conj(wavefunctions) * H_psi, dim=-1)
        norm = torch.sum(torch.abs(wavefunctions) ** 2, dim=-1)
        eps = torch.finfo(wavefunctions.real.dtype).eps
        return (numerator / norm.clamp_min(eps)).real


class NormalModes(nn.Module):
    def __init__(self, state_dim: int, num_modes: int):
        super().__init__()
        self.real = nn.Parameter(torch.randn(state_dim, num_modes) * 0.02)
        self.imag = nn.Parameter(torch.randn(state_dim, num_modes) * 0.02)

    def forward(self) -> torch.Tensor:
        return torch.complex(self.real, self.imag)


class ModeExcitations(nn.Module):
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

    def collapse(self, modes: torch.Tensor) -> torch.Tensor:
        return torch.abs(modes.sum(dim=1)).pow(2)

    def forward(
        self,
        observation: torch.Tensor,
        psi_latent: torch.Tensor,
        time: torch.Tensor,
        external_potential: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.is_periodic:
            obs_scaled = observation * torch.cos(
                2.0 * torch.pi * self.observation_freq * time
            ).pow(2)
        else:
            obs_scaled = observation

        psi = self.superposition(obs_scaled, psi_latent)
        modes = self.propagator(psi, external_potential, self.dt)
        next_latent = modes

        if self.is_periodic:
            next_time = (time + self.dt) % self.observation_interval
        else:
            next_time = time

        return modes, self.collapse(modes), next_latent, next_time


class PhasorStack(nn.Module):
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
        layer_latents: list[torch.Tensor],
        layer_times: list[torch.Tensor],
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor]],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        results = []
        next_latents = []
        next_times = []
        external_potential = None

        for p_idx, phasor in enumerate(self.phasors):
            modes, collapsed, next_psi, next_t = phasor(
                observation=observation,
                psi_latent=layer_latents[p_idx],
                time=layer_times[p_idx],
                external_potential=external_potential,
            )
            results.append((modes, collapsed))
            next_latents.append(next_psi)
            next_times.append(next_t)
            external_potential = collapsed

        return results, next_latents, next_times


class PhasorNet(nn.Module):
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

    def forward(
        self,
        observation: torch.Tensor,
        state: PhasorState | None = None,
    ) -> tuple[
        torch.Tensor,
        list[list[tuple[torch.Tensor, torch.Tensor]]],
        PhasorState,
    ]:
        if state is None:
            state = PhasorState.create_zeros(
                self.config,
                batch_size=observation.shape[0],
                device=observation.device,
            )

        stack_observation = observation
        all_results = []
        next_latents_all = []
        next_times_all = []

        for layer, stack in enumerate(self.stacks):
            results, next_layer_latents, next_layer_times = stack(
                observation=stack_observation,
                layer_latents=state.latents[layer],
                layer_times=state.times[layer],
            )
            all_results.append(results)
            next_latents_all.append(next_layer_latents)
            next_times_all.append(next_layer_times)
            stack_observation = results[-1][1]

        final_collapse = all_results[-1][-1][1]
        final_observable = self.observable_projection(final_state_val)
        next_state = PhasorState(latents=next_latents_all, times=next_times_all)

        return final_observable, all_results, next_state
