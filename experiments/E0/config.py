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
    num_layers: int = 2
    
    observation_freqs: tuple[float, ...] = (1/32, 1/16, 0.0)
    # Observation dimension at each hierarchical layer.
    observable_dims: tuple[int, ...] = (64, 128)
    # State dimension of each layer.
    state_dims: tuple[int, ...] = (128, 128)
    # Number of modes in each Phasor.
    num_modes: tuple[int, ...] = (8, 8)
    # Hidden dimension for mode excitation.
    hidden_dims: tuple[int, ...] = (256, 256)



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
            num_layers=self.num_layers,
            observation_freqs=self.observation_freqs,
            observable_dims=self.observable_dims,
            state_dims=self.state_dims,
            num_modes=self.num_modes,
            hidden_dims=self.hidden_dims,
            dt=self.dt,
        )
