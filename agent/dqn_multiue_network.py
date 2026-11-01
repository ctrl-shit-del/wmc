"""
dqn_multiue_network.py
Multi-UE DQN with dual-head architecture.

State:  48D  (5 UEs x 9 features each + 3 global)
        Per UE: [sinr_norm, x_norm, y_norm,
                 vx_norm, vy_norm, prev_beam_norm,
                 distance_norm, speed_norm, sinr_h1_norm]
        Global: [min_sinr_norm, fairness, step_norm]

Action: Two-head decomposed action
        Head 1 → target UE   (5 outputs)
        Head 2 → beam index  (32 outputs)
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path


NUM_UE    = 5
NUM_BEAMS = 32
STATE_DIM = NUM_UE * 9 + 3   # 48
ACTION_DIM_UE   = NUM_UE      # 5
ACTION_DIM_BEAM = NUM_BEAMS   # 32


class MultiUeDQN(nn.Module):
    """
    Dual-head DQN for multi-UE beam management.
    Shared feature extractor → UE selection head
                             → Beam selection head
    """
    def __init__(self,
                 state_dim: int = STATE_DIM,
                 num_ue: int    = NUM_UE,
                 num_beams: int = NUM_BEAMS):
        super().__init__()
        self.num_ue    = num_ue
        self.num_beams = num_beams

        # Shared encoder: extracts joint UE context
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Head 1: Q-values for UE selection
        self.ue_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_ue),
        )

        # Head 2: Q-values for beam selection
        # Takes shared features + UE one-hot context
        self.beam_head = nn.Sequential(
            nn.Linear(256 + num_ue, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_beams),
        )

    def forward(self,
                x: torch.Tensor,
                ue_mask: torch.Tensor   = None,
                beam_mask: torch.Tensor = None
                ) -> tuple:
        """
        Returns:
            q_ue   : (batch, num_ue)    Q-values for UE selection
            q_beam : (batch, num_beams) Q-values for beam selection
        """
        feat = self.shared(x)

        # UE Q-values
        q_ue = self.ue_head(feat)
        if ue_mask is not None:
            q_ue = q_ue.masked_fill(~ue_mask,
                                     float('-inf'))

        # Beam Q-values conditioned on UE selection
        ue_context = torch.softmax(q_ue.detach(), dim=-1)
        beam_input = torch.cat([feat, ue_context], dim=-1)
        q_beam = self.beam_head(beam_input)
        if beam_mask is not None:
            q_beam = q_beam.masked_fill(~beam_mask,
                                         float('-inf'))

        return q_ue, q_beam


class MultiUeActionMasker:
    """
    Position-guided action masking for multiple UEs.
    For each UE, restricts beam candidates to
    K geometrically nearest beams.
    """
    def __init__(self,
                 num_beams: int = NUM_BEAMS,
                 k: int         = 8):
        self.num_beams = num_beams
        self.k         = k
        # Beam azimuths: uniform 0–360°
        self.azimuths  = np.linspace(
            0, 2 * np.pi, num_beams, endpoint=False)

    def get_beam_mask(self,
                      ue_x: float,
                      ue_y: float) -> torch.Tensor:
        """Get K-nearest beam mask for a single UE."""
        predicted_az = np.arctan2(ue_y, ue_x) % (2*np.pi)
        diffs = np.abs(self.azimuths - predicted_az)
        diffs = np.minimum(diffs, 2*np.pi - diffs)
        valid = np.argsort(diffs)[:self.k]
        mask  = torch.zeros(self.num_beams,
                             dtype=torch.bool)
        mask[valid] = True
        return mask

    def get_ue_beam_mask(self,
                          ue_positions: list,
                          target_ue: int
                          ) -> torch.Tensor:
        """Get beam mask for the selected target UE."""
        x, y = ue_positions[target_ue]
        return self.get_beam_mask(x * 200, y * 200)


def build_state_vector(ue_data: list,
                        step: int,
                        total_steps: int = 2800
                        ) -> np.ndarray:
    """
    Build 48D state vector from 5 UEs.

    ue_data: list of 5 dicts with keys:
        sinr_db, x_norm, y_norm, vx_norm, vy_norm,
        prev_beam, sinr_history

    Returns: float32 array of shape (48,)
    """
    SINR_MIN, SINR_MAX = -20.0, 100.0
    SINR_RANGE = SINR_MAX - SINR_MIN

    features = []
    sinrs = [d['sinr_db'] for d in ue_data]

    for d in ue_data:
        sinr_norm = np.clip(
            (d['sinr_db'] - SINR_MIN) / SINR_RANGE,
            0, 1)
        dist_norm = np.sqrt(d['x_norm']**2 +
                             d['y_norm']**2)
        speed_norm = np.sqrt(d['vx_norm']**2 +
                              d['vy_norm']**2)
        sinr_h1 = np.clip(
            (d.get('sinr_h1', d['sinr_db'])
             - SINR_MIN) / SINR_RANGE, 0, 1)

        features.extend([
            sinr_norm,
            d['x_norm'],
            d['y_norm'],
            d['vx_norm'],
            d['vy_norm'],
            d['prev_beam'] / 32.0,
            dist_norm,
            speed_norm,
            sinr_h1,
        ])

    # Global features
    min_sinr_norm = np.clip(
        (min(sinrs) - SINR_MIN) / SINR_RANGE, 0, 1)
    mean_sinr     = np.mean(sinrs)
    fairness      = (min(sinrs) / (mean_sinr + 1e-6)
                     if mean_sinr > 0 else 0.0)
    step_norm     = step / total_steps

    features.extend([min_sinr_norm, fairness, step_norm])

    return np.array(features, dtype=np.float32)


def jains_fairness(sinr_list: list) -> float:
    """Jain's fairness index. 1.0 = perfectly fair."""
    x = np.array(sinr_list)
    if x.sum() == 0:
        return 0.0
    return float((x.sum()**2) /
                  (len(x) * (x**2).sum() + 1e-9))


if __name__ == "__main__":
    print("=== MultiUeDQN Unit Test ===")

    net = MultiUeDQN(state_dim=STATE_DIM,
                      num_ue=NUM_UE,
                      num_beams=NUM_BEAMS)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"State dim:    {STATE_DIM}")
    print(f"UE actions:   {ACTION_DIM_UE}")
    print(f"Beam actions: {ACTION_DIM_BEAM}")
    print(f"Parameters:   {total_params:,}")

    # Test forward pass
    dummy = torch.randn(4, STATE_DIM)
    q_ue, q_beam = net(dummy)
    print(f"\nForward pass (batch=4):")
    print(f"  q_ue   shape: {q_ue.shape}   "
          f"range: [{q_ue.min():.3f}, {q_ue.max():.3f}]")
    print(f"  q_beam shape: {q_beam.shape}  "
          f"range: [{q_beam.min():.3f}, {q_beam.max():.3f}]")

    # Test action masking
    masker = MultiUeActionMasker(num_beams=32, k=8)
    ue_positions = [
        (0.5, 0.3), (-0.4, 0.6),
        (0.2, -0.7), (-0.6, -0.2), (0.1, 0.8)
    ]
    for i, (x, y) in enumerate(ue_positions):
        mask  = masker.get_beam_mask(x*200, y*200)
        print(f"  UE{i} ({x:.1f},{y:.1f}): "
              f"{mask.sum().item()} active beams → "
              f"{mask.nonzero().squeeze().tolist()}")

    # Test state vector builder
    ue_data = [{
        'sinr_db': 85.0 + i*2, 'x_norm': 0.3+i*0.1,
        'y_norm': 0.2+i*0.05, 'vx_norm': 0.1,
        'vy_norm': 0.0, 'prev_beam': i*3,
        'sinr_h1': 84.0+i*2
    } for i in range(NUM_UE)]
    state = build_state_vector(ue_data, step=100)
    print(f"\nState vector: shape={state.shape} "
          f"range=[{state.min():.3f},{state.max():.3f}]")

    # Test Jain's fairness
    sinrs = [85.0, 91.0, 92.0, 76.0, 82.0]
    print(f"\nJain's fairness {sinrs}: "
          f"{jains_fairness(sinrs):.4f}")
    print(f"Perfect fairness  [85,85,85,85,85]: "
          f"{jains_fairness([85]*5):.4f}")

    print("\nMultiUeDQN test PASSED")
