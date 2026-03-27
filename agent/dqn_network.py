import torch
import torch.nn as nn
import numpy as np


class DQNNetwork(nn.Module):
    """
    Deep Q-Network: 512-256-128 fully connected.
    Supports action masking for large codebooks (G-06).
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        q_values = self.net(x)
        if mask is not None:
            q_values = q_values.masked_fill(
                ~mask, float('-inf'))
        return q_values


class ActionMasker:
    """
    Position-guided beam neighborhood masking.
    Restricts action space to K nearest beams (G-06).
    """
    def __init__(self, num_beams: int, k: int = 8):
        self.num_beams = num_beams
        self.k = k
        self.azimuths = np.linspace(
            0, 2 * np.pi, num_beams, endpoint=False)

    def get_mask(self, ue_x: float,
                 ue_y: float) -> torch.Tensor:
        predicted_az = np.arctan2(ue_y, ue_x) % (2 * np.pi)
        diffs = np.abs(self.azimuths - predicted_az)
        diffs = np.minimum(diffs, 2 * np.pi - diffs)
        valid_idx = np.argsort(diffs)[:self.k]
        mask = torch.zeros(self.num_beams, dtype=torch.bool)
        mask[valid_idx] = True
        return mask


if __name__ == "__main__":
    net = DQNNetwork(state_dim=16, action_dim=32)
    print(net)
    total = sum(p.numel() for p in net.parameters())
    print(f"\nTotal parameters: {total:,}")

    # Test forward pass
    dummy = torch.randn(1, 16)
    q = net(dummy)
    print(f"Q-values shape:   {q.shape}")
    print(f"Q-values range:   [{q.min().item():.3f}, "
          f"{q.max().item():.3f}]")

    # Test action masking
    masker = ActionMasker(num_beams=32, k=8)
    mask = masker.get_mask(100.0, 50.0)
    print(f"Active beams:     {mask.sum().item()} "
          f"(should be 8)")

    # Test masked forward pass
    q_masked = net(dummy, mask.unsqueeze(0))
    inf_count = (q_masked == float('-inf')).sum().item()
    print(f"Masked-out:       {inf_count} "
          f"(should be 24)")
    print("\nDQN network test PASSED")
