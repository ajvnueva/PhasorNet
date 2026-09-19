from dataclasses import dataclass
from phasornet import PhasorNetConfig

@dataclass
class E0Config:
    # --- Physical & Signal Parameters ---
    feature_dim: int = 64
    dt: float = 0.05
    seq_len_range: tuple = (64, 64)
    num_components_range: tuple = (2, 6)

    # --- PhasorNet Architecture ---
    num_phasors: int = 3
    observation_intervals: tuple[int, ...] = (32, 16, 1)

    observation_dims: tuple[int, ...] = (64, 64, 64)
    state_dims: tuple[int, ...] = (128, 128, 128)
    num_modes: tuple[int, ...] = (8, 8, 8)
    hidden_dims: tuple[int, ...] = (256, 256, 256)

    num_layers: int = 2

    # --- Training Hyperparameters ---
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5
    train_samples: int = 10000
    val_samples: int = 1000

    # --- Visualization & Evaluation ---
    animate_eval: bool = True
    anim_filename: str = "phasornet_adaptation.gif"
    anim_fps: int = 15

    def to_phasornet_config(self) -> PhasorNetConfig:
        return PhasorNetConfig(
            num_phasors=self.num_phasors,
            observation_intervals=self.observation_intervals,
            observation_dims=self.observation_dims,
            state_dims=self.state_dims,
            num_modes=self.num_modes,
            hidden_dims=self.hidden_dims,
            num_layers=self.num_layers,
            dt=self.dt,
        )
