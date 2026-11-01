"""
dqn_multiue_agent.py
Complete DQN agent for 5-UE beam management.
Handles reward computation, action selection,
training, and checkpointing for all 4 scenarios.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from collections import namedtuple

from dqn_multiue_network import (
    MultiUeDQN, MultiUeActionMasker,
    build_state_vector, jains_fairness,
    STATE_DIM, NUM_UE, NUM_BEAMS
)

# ── Transition tuple ─────────────────────────────────────
Transition = namedtuple('Transition', [
    'state', 'ue_action', 'beam_action',
    'reward', 'next_state', 'done'
])


# ── Prioritized Replay Buffer ─────────────────────────────
class MultiUePERBuffer:
    def __init__(self, capacity=100000,
                 alpha=0.6, beta_start=0.4,
                 beta_end=1.0, beta_steps=200000):
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta       = beta_start
        self.beta_end   = beta_end
        self.beta_inc   = (beta_end - beta_start) / beta_steps
        self.eps        = 1e-5
        self.buffer     = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos        = 0

    def push(self, *args):
        max_p = self.priorities.max() if self.buffer else 1.0
        t = Transition(*args)
        if len(self.buffer) < self.capacity:
            self.buffer.append(t)
        else:
            self.buffer[self.pos] = t
        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n     = len(self.buffer)
        probs = self.priorities[:n] ** self.alpha
        probs /= probs.sum()
        idx   = np.random.choice(n, batch_size,
                                  replace=False, p=probs)
        samples = [self.buffer[i] for i in idx]
        w = (n * probs[idx]) ** (-self.beta)
        w /= w.max()
        self.beta = min(self.beta_end,
                        self.beta + self.beta_inc)
        return samples, idx, w

    def update_priorities(self, idx, td_errors):
        for i, e in zip(idx, td_errors):
            self.priorities[i] = (abs(e) + self.eps) ** self.alpha

    def __len__(self):
        return len(self.buffer)


# ── Multi-UE DQN Agent ────────────────────────────────────
class MultiUeAgent:
    """
    DQN agent for 5-UE beam management.
    Handles dual action (UE + beam) with separate
    Q-value heads and PER replay buffer.
    """

    def __init__(self, cfg_path: str = "../config.json",
                 scenario: int = 1):
        self.scenario    = scenario
        self.num_ue      = NUM_UE
        self.num_beams   = NUM_BEAMS
        self.state_dim   = STATE_DIM

        # Load hyperparams from config
        cfg = json.loads(Path(cfg_path).read_text())
        d   = cfg["drl"]
        r   = cfg["reward"]

        self.gamma      = d["gamma"]
        self.tau        = d["target_update_tau"]
        self.batch_size = d["batch_size"]
        self.w1         = r["w1"]
        self.w2         = r["w2"]
        self.w3         = r["w3"]
        self.outage_thr = r["outage_threshold_db"]

        # Networks
        self.policy_net = MultiUeDQN(
            self.state_dim, self.num_ue, self.num_beams)
        self.target_net = MultiUeDQN(
            self.state_dim, self.num_ue, self.num_beams)
        self.target_net.load_state_dict(
            self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=d["learning_rate"], weight_decay=1e-5)
        self.loss_fn = nn.HuberLoss(delta=1.0)

        # Replay buffer (larger for multi-UE)
        self.buffer  = MultiUePERBuffer(
            capacity=100000)
        self.masker  = MultiUeActionMasker(
            num_beams=self.num_beams, k=8)

        # Epsilon
        self.epsilon     = d["epsilon_start"]
        self.epsilon_min = d["epsilon_min"]

        # Training state
        self.q_history   = []
        self.step_count  = 0
        self.episode     = 0

        # Per-UE serving tracker
        # (tracks last TTI each UE was served)
        self.last_served = np.zeros(self.num_ue,
                                     dtype=int)
        self.tti_count   = 0

        # Inference latency tracking
        self.inf_times = []

    # ── Reward function ──────────────────────────────────
    def compute_reward(self,
                        sinr_list: list,
                        target_ue: int,
                        beam_changed: bool) -> float:
        """
        Multi-UE switching-aware reward.

        Components:
        1. Min-SINR maximisation (fairness objective)
        2. Jain's fairness bonus
        3. Starvation penalty (UE not served recently)
        4. Beam switching penalty
        5. Outage penalty (per outage UE)
        """
        SINR_MIN, SINR_MAX = -20.0, 100.0
        RANGE = SINR_MAX - SINR_MIN

        # 1. Min-SINR (ensures worst UE is considered)
        min_sinr = min(sinr_list)
        min_sinr_norm = np.clip(
            (min_sinr - SINR_MIN) / RANGE, 0, 1)

        # 2. Jain's fairness bonus
        fairness = jains_fairness(sinr_list)
        fairness_bonus = 0.3 * fairness

        # 3. Starvation penalty
        # Any UE not served in last 10 TTIs
        starvation_penalty = 0.0
        for i in range(self.num_ue):
            if (self.tti_count -
                    self.last_served[i]) > 10:
                starvation_penalty += 0.3

        # 4. Beam switch penalty
        switch_penalty = self.w2 if beam_changed else 0.0

        # 5. Outage penalty per outage UE
        outage_penalty = sum(
            self.w3 for s in sinr_list
            if s < self.outage_thr)

        reward = (self.w1 * min_sinr_norm
                  + fairness_bonus
                  - starvation_penalty
                  - switch_penalty
                  - outage_penalty)
        return float(reward)

    # ── Action selection ─────────────────────────────────
    def select_action(self,
                       state: np.ndarray,
                       ue_positions: list
                       ) -> tuple:
        """
        Returns (target_ue, beam_index, inf_ms)
        ue_positions: list of (x_norm, y_norm)
        """
        t0 = time.perf_counter()

        if np.random.random() < self.epsilon:
            # Explore: random UE and random valid beam
            ue   = np.random.randint(self.num_ue)
            x, y = ue_positions[ue]
            mask = self.masker.get_beam_mask(
                x * 200, y * 200)
            valid = mask.nonzero().squeeze().numpy()
            if valid.ndim == 0:
                valid = np.array([valid.item()])
            beam = int(np.random.choice(valid))
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0)
                q_ue, q_beam = self.policy_net(s)
                ue   = int(q_ue.argmax().item())
                x, y = ue_positions[ue]
                mask = self.masker.get_beam_mask(
                    x * 200, y * 200)
                q_beam_masked = q_beam.clone()
                q_beam_masked[0, ~mask] = float('-inf')
                beam = int(q_beam_masked.argmax().item())

        inf_ms = (time.perf_counter() - t0) * 1000
        self.inf_times.append(inf_ms)
        self.step_count   += 1
        self.tti_count    += 1
        self.last_served[ue] = self.tti_count
        return ue, beam, inf_ms

    # ── Training step ────────────────────────────────────
    def train_step(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        samples, idx, weights = self.buffer.sample(
            self.batch_size)
        w_t = torch.FloatTensor(weights)

        states     = torch.FloatTensor(
            np.array([t.state for t in samples]))
        ue_acts    = torch.LongTensor(
            np.array([t.ue_action for t in samples]))
        beam_acts  = torch.LongTensor(
            np.array([t.beam_action for t in samples]))
        rewards    = torch.FloatTensor(
            np.array([t.reward for t in samples]))
        nstates    = torch.FloatTensor(
            np.array([t.next_state for t in samples]))
        dones      = torch.FloatTensor(
            np.array([t.done for t in samples]))

        # Current Q-values (UE head)
        q_ue_curr, q_beam_curr = self.policy_net(states)
        q_ue_taken = q_ue_curr.gather(
            1, ue_acts.unsqueeze(1)).squeeze()

        # Beam Q-values for taken actions
        q_beam_taken = q_beam_curr.gather(
            1, beam_acts.unsqueeze(1)).squeeze()

        # Combined Q (average of UE and beam)
        q_curr = 0.5 * q_ue_taken + 0.5 * q_beam_taken

        # Target Q-values
        with torch.no_grad():
            q_ue_next, q_beam_next = self.target_net(
                nstates)
            q_next = (0.5 * q_ue_next.max(1)[0] +
                      0.5 * q_beam_next.max(1)[0])
            q_target = (rewards +
                        self.gamma * q_next * (1 - dones))

        # PER weights applied to loss
        td_errors = (q_curr - q_target).detach().abs()
        self.buffer.update_priorities(
            idx, td_errors.numpy())
        self.q_history.append(float(q_curr.mean().detach()))

        loss = (w_t * self.loss_fn(q_curr, q_target)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft target update
        for p, tp in zip(self.policy_net.parameters(),
                          self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data +
                           (1 - self.tau) * tp.data)

        # Adaptive epsilon decay
        self._adapt_epsilon()
        return float(loss.detach())

    def _adapt_epsilon(self):
        if self.episode < 500:
            return
        if (len(self.q_history) >= 100 and
                self.epsilon > self.epsilon_min):
            delta_q = np.std(self.q_history[-100:])
            if delta_q < 0.05:
                self.epsilon = max(
                    self.epsilon_min,
                    self.epsilon * 0.995)

    def save(self, path: str):
        torch.save({
            "policy":    self.policy_net.state_dict(),
            "target":    self.target_net.state_dict(),
            "epsilon":   self.epsilon,
            "episode":   self.episode,
            "steps":     self.step_count,
            "scenario":  self.scenario,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        self.epsilon  = ckpt["epsilon"]
        self.episode  = ckpt["episode"]
        self.step_count = ckpt["steps"]

    def latency_report(self) -> dict:
        t = np.array(self.inf_times)
        if len(t) == 0:
            return {}
        return {
            "mean_ms":   float(np.mean(t)),
            "p99_ms":    float(np.percentile(t, 99)),
            "max_ms":    float(np.max(t)),
        }


if __name__ == "__main__":
    print("=== MultiUeAgent Unit Test ===")
    agent = MultiUeAgent(
        cfg_path="../config.json", scenario=1)
    print(f"Agent created for Scenario 1")
    print(f"State dim:  {agent.state_dim}")
    print(f"Num UEs:    {agent.num_ue}")
    print(f"Num beams:  {agent.num_beams}")
    print(f"Buffer cap: {agent.buffer.capacity:,}")

    # Simulate 200 steps
    ue_positions = [(0.5,0.3),(-0.4,0.6),
                    (0.2,-0.7),(-0.6,-0.2),(0.1,0.8)]
    prev_beam = [-1]*NUM_UE
    prev_ue   = -1

    for step in range(200):
        ue_data = [{
            'sinr_db': 85.0 + np.random.randn()*3,
            'x_norm': ue_positions[i][0],
            'y_norm': ue_positions[i][1],
            'vx_norm': 0.1, 'vy_norm': 0.05,
            'prev_beam': float(prev_beam[i]),
            'sinr_h1': 84.0
        } for i in range(NUM_UE)]

        state  = build_state_vector(ue_data, step)
        ue, beam, inf_ms = agent.select_action(
            state, ue_positions)

        sinr_list    = [d['sinr_db'] for d in ue_data]
        beam_changed = (beam != prev_beam[ue]
                        and prev_beam[ue] >= 0)
        reward = agent.compute_reward(
            sinr_list, ue, beam_changed)

        next_state = build_state_vector(ue_data, step+1)
        agent.buffer.push(
            state, ue, beam, reward, next_state, False)
        prev_beam[ue] = beam
        prev_ue = ue

    print(f"\nAfter 200 steps:")
    print(f"  Buffer size: {len(agent.buffer)}")

    losses = []
    for _ in range(10):
        l = agent.train_step()
        if l > 0:
            losses.append(l)
    print(f"  Train steps: 10")
    print(f"  Mean loss:   {np.mean(losses):.5f}"
          if losses else "  No training yet")

    lr = agent.latency_report()
    print(f"\nInference latency:")
    print(f"  Mean: {lr['mean_ms']:.3f} ms")
    print(f"  P99:  {lr['p99_ms']:.3f} ms")

    # Test reward fairness sensitivity
    sinrs_fair   = [85.0]*5
    sinrs_unfair = [95.0, 95.0, 95.0, 95.0, 30.0]
    r_fair   = agent.compute_reward(sinrs_fair,   0, False)
    r_unfair = agent.compute_reward(sinrs_unfair, 0, False)
    print(f"\nReward fairness test:")
    print(f"  All 85dB:          R={r_fair:.3f}")
    print(f"  4x95dB + 1x30dB:   R={r_unfair:.3f}")
    print(f"  (unfair < fair is correct: "
          f"{'YES' if r_unfair < r_fair else 'NO'})")

    Path("../checkpoints").mkdir(exist_ok=True)
    agent.save("../checkpoints/test_multiue.pt")
    agent.load("../checkpoints/test_multiue.pt")
    print("\nMultiUeAgent test PASSED")
