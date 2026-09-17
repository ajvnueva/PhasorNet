

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# =====================================================================
# 1. Superposed Harmonic Wave Dataset Generator
# =====================================================================

class SuperposedSineDataset(Dataset):
    """
    Generates time-evolving wave sequences formed by a superposition of sines.
    
    - Feature Space (D dimensions): Spatial domain x in [0, 2π).
    - Superposition: Linear combination of K harmonic modes with unique spatial frequencies,
      phase velocities, and amplitude envelope modulations.
    - Variable Sequence Length (T): Sampled randomly per item.
    - Prediction Target: Shifted by 1 timestep (Y_t = X_{t+1}).
    """

    def __init__(
        self,
        num_samples: int = 1000,
        feature_dim: int = 64,
        seq_len_range: tuple = (30, 80),
        num_components_range: tuple = (2, 4),
        dt: float = 0.05,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.seq_len_range = seq_len_range
        self.num_components_range = num_components_range
        self.dt = dt

        # Discretize feature space across [0, 2π)
        self.x = torch.linspace(
            0, 2 * math.pi * (1.0 - 1.0 / feature_dim), feature_dim
        )

    def __len__(self):
        return self.num_samples

    def _generate_single_sequence(self, seq_len: int):
        # Sample number of superposed components K
        K = torch.randint(
            self.num_components_range[0],
            self.num_components_range[1] + 1,
            (1,),
        ).item()

        # Randomize parameters for each superposed component m
        k_spatial = torch.randint(1, 4, (K,)).float()  # Spatial harmonics (1, 2, or 3 periods)
        phi_0 = torch.rand(K) * 2 * math.pi             # Initial phase offsets
        base_omega = 1.0 + torch.rand(K) * 2.0         # Base phase velocities
        mod_freq = 0.5 + torch.rand(K) * 1.5           # Phase acceleration rates
        amp_base = 0.5 + torch.rand(K) * 0.5           # Base component amplitudes
        amp_freq = 0.2 + torch.rand(K) * 0.8           # Amplitude envelope frequencies

        sequence = []
        current_phases = phi_0.clone()

        for step in range(seq_len + 1):
            t = step * self.dt

            # Dynamic amplitude and phase velocity update for all components
            amps = amp_base * (1.0 + 0.4 * torch.sin(amp_freq * t))
            omegas_t = base_omega + 0.5 * torch.sin(mod_freq * t)
            current_phases += omegas_t * self.dt

            # Evaluate each mode: A_m(t) * sin(k_m * x - phi_m(t))
            # [K, 1] * sin([K, 1] * [1, D] - [K, 1]) -> [K, D]
            modes = amps.unsqueeze(1) * torch.sin(
                k_spatial.unsqueeze(1) * self.x.unsqueeze(0) - current_phases.unsqueeze(1)
            )

            # Superposition (Sum across all K components)
            superposed_wave = modes.sum(dim=0)  # [D]
            sequence.append(superposed_wave)

        sequence = torch.stack(sequence, dim=0)  # [T + 1, D]

        # 1-timestep prediction offset
        inputs = sequence[:-1]  # [T, D]
        targets = sequence[1:]  # [T, D]

        return inputs, targets

    def __getitem__(self, idx):
        seq_len = torch.randint(
            self.seq_len_range[0], self.seq_len_range[1] + 1, (1,)
        ).item()
        return self._generate_single_sequence(seq_len)


# =====================================================================
# 2. Animation Renderer for Superposed Wave Evolutions
# =====================================================================

def animate_superposition(
    inputs, targets, dt=0.05, save_path="superposition_evolution.gif"
):
    """
    Renders an animated visualization of the superposed wave evolution.
    """
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()

    T, D = inputs.shape
    x = np.linspace(0, 2 * np.pi * (1.0 - 1.0 / D), D)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    (line_inp,) = ax.plot(
        x, inputs[0], color="#1E88E5", lw=2.5, label="Superposed Input Wave (t)"
    )
    (line_tgt,) = ax.plot(
        x,
        targets[0],
        color="#D81B60",
        linestyle="--",
        lw=2.0,
        label="Target Evolution (t+1)",
    )

    y_min, y_max = inputs.min() * 1.25, inputs.max() * 1.25
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, 2 * np.pi)

    ax.set_title("Superposed Wave Evolution (t = 0.00s)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Spatial Feature Axis (x ∈ [0, 2π))", fontsize=10)
    ax.set_ylabel("Wave Amplitude W(x, t)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    def update(frame):
        line_inp.set_ydata(inputs[frame])
        line_tgt.set_ydata(targets[frame])
        ax.set_title(
            f"Superposed Wave Evolution (t = {frame * dt:.2f}s | Step {frame}/{T-1})",
            fontweight="bold",
            fontsize=12,
        )
        return line_inp, line_tgt

    ani = animation.FuncAnimation(fig, update, frames=T, interval=60, blit=True)
    ani.save(save_path, writer="pillow", fps=18)
    plt.close(fig)
    print(f"Saved superposition animation to: '{save_path}'")


if __name__ == "__main__":
    # Instantiate dataset with 2 to 4 superposed sine modes
    dataset = SuperposedSineDataset(
        num_samples=100,
        feature_dim=64,
        seq_len_range=(40, 70),
        num_components_range=(2, 4),
        dt=0.05,
    )

    sample_inputs, sample_targets = dataset[0]
    print(f"Sample generated with shape: {sample_inputs.shape}")

    # Generate and save evolution animation
    animate_superposition(
        sample_inputs, sample_targets, dt=0.05, save_path="superposition_evolution.gif"
    )
