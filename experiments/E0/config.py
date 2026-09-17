from dataclasses import dataclass
from phasornet import PhasorNetConfig


@dataclass
class E0Config:
    # --- Physical & Signal Parameters ---
    feature_dim: int = 64        # Spatial resolution of sines (x-grid)
    dt: float = 0.05             # Time step interval
    seq_len_range: tuple = (30, 80)
    num_components_range: tuple = (2, 4)

    # --- Model Architecture ---
    state_dim: int = 128
    num_modes: int = 16
    hidden_dim: int = 256
    num_layers: int = 2

    # --- Training Hyperparameters ---
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 20
    train_samples: int = 10000
    val_samples: int = 1000

    def to_phasornet_config(self) -> PhasorNetConfig:
        """Converts experiment settings directly to PhasorNetConfig,

        guaranteeing observation_dim and dt stay perfectly aligned with the dataset.
        """
        return PhasorNetConfig(
            observation_dim=self.feature_dim,  # Auto-aligned to dataset!
            state_dim=self.state_dim,
            num_modes=self.num_modes,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dt=self.dt,                          # Auto-aligned to dataset!
        )
