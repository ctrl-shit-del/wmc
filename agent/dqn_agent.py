import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from dqn_network import DQNNetwork, ActionMasker
from per_buffer import PERBuffer


class DQNAgent:
    def __init__(self, cfg_path: str = "../config.json"):
        cfg = json.loads(Path(cfg_path).read_text())
        d   = cfg["drl"]
        r   = cfg["reward"]
        cb  = cfg["codebook"]

        self.state_dim  = d["state_dim"]
        self.num_beams  = cb["num_beams"]
        self.gamma      = d["gamma"]
        self.tau        = d["target_update_tau"]
        self.batch_size = d["batch_size"]
        self.w1         = r["w1"]
        self.w2         = r["w2"]
        self.w3         = r["w3"]
        self.outage_thr = r["outage_threshold_db"]

        # Networks
        self.policy_net = DQNNetwork(
            self.state_dim, self.num_beams)
        self.target_net = DQNNetwork(
            self.state_dim, self.num_beams)
        self.target_net.load_state_dict(
            self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=d["learning_rate"],
            weight_decay=1e-5)
        self.loss_fn = nn.HuberLoss(delta=1.0)

        self.buffer  = PERBuffer(
            capacity=d["buffer_capacity"])
        self.masker  = ActionMasker(
            self.num_beams, cb["action_mask_k"])

        # Epsilon (G-09 adaptive decay)
        self.epsilon     = d["epsilon_start"]
        self.epsilon_min = d["epsilon_min"]
        self.q_history   = []
        self.step_count  = 0
        self.episode     = 0

        # SINR normalisation constants
        self.sinr_min = -20.0
        self.sinr_max =  40.0

        # Inference latency tracking (G-16)
        self.inference_times = []

    def normalise_sinr(self, sinr_db: float) -> float:
        return float(np.clip(
            (sinr_db - self.sinr_min) /
            (self.sinr_max - self.sinr_min), 0.0, 1.0))

    def compute_reward(self, sinr_db: float,
                       beam_changed: bool) -> float:
        """Switching-aware reward function (G-07)."""
        sinr_norm = self.normalise_sinr(sinr_db)
        switch_p  = self.w2 if beam_changed else 0.0
        outage_p  = self.w3 \
                    if sinr_db < self.outage_thr else 0.0
        return self.w1 * sinr_norm - switch_p - outage_p

    def select_action(self, state: np.ndarray,
                      ue_x: float,
                      ue_y: float) -> tuple:
        mask = self.masker.get_mask(ue_x, ue_y)
        t0   = time.perf_counter()

        if np.random.random() < self.epsilon:
            valid = mask.nonzero().squeeze().numpy()
            if valid.ndim == 0:
                valid = np.array([valid.item()])
            action = int(np.random.choice(valid))
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0)
                q = self.policy_net(s, mask.unsqueeze(0))
                action = int(q.argmax().item())

        inf_ms = (time.perf_counter() - t0) * 1000
        self.inference_times.append(inf_ms)
        self.step_count += 1
        return action, inf_ms

    def _adapt_epsilon(self):
        """Adaptive decay: only reduce when Q converges (G-09)."""
        if self.episode < 500:
            return
        if len(self.q_history) >= 100:
            delta_q = np.std(self.q_history[-100:])
            if (delta_q < 0.05 and
                    self.epsilon > self.epsilon_min):
                self.epsilon = max(
                    self.epsilon_min,
                    self.epsilon * 0.995)

    def train_step(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        samples, indices, weights = \
            self.buffer.sample(self.batch_size)
        w_t = torch.FloatTensor(weights)

        states  = torch.FloatTensor(
            np.array([t.state      for t in samples]))
        actions = torch.LongTensor(
            np.array([t.action     for t in samples]))
        rewards = torch.FloatTensor(
            np.array([t.reward     for t in samples]))
        nstates = torch.FloatTensor(
            np.array([t.next_state for t in samples]))
        dones   = torch.FloatTensor(
            np.array([t.done       for t in samples]))

        q_curr = self.policy_net(states)\
                     .gather(1, actions.unsqueeze(1))\
                     .squeeze()

        with torch.no_grad():
            q_next   = self.target_net(nstates).max(1)[0]
            q_target = rewards + \
                       self.gamma * q_next * (1 - dones)

        td_errors = (q_curr - q_target)\
                        .detach().abs().numpy()
        self.buffer.update_priorities(indices, td_errors)
        self.q_history.append(float(q_curr.mean().detach()))

        loss = (w_t * self.loss_fn(
                    q_curr, q_target)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Soft target update
        for p, tp in zip(
                self.policy_net.parameters(),
                self.target_net.parameters()):
            tp.data.copy_(
                self.tau * p.data +
                (1 - self.tau) * tp.data)

        self._adapt_epsilon()
        return float(loss.detach())

    def save(self, path: str):
        torch.save({
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "epsilon":      self.epsilon,
            "episode":      self.episode,
            "step_count":   self.step_count,
        }, path)
        print(f"Checkpoint saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path)
        self.policy_net.load_state_dict(
            ckpt["policy_state"])
        self.target_net.load_state_dict(
            ckpt["target_state"])
        self.epsilon    = ckpt["epsilon"]
        self.episode    = ckpt["episode"]
        self.step_count = ckpt["step_count"]
        print(f"Checkpoint loaded: {path}")

    def latency_report(self) -> dict:
        t = np.array(self.inference_times)
        return {
            "mean_ms":   float(np.mean(t)),
            "median_ms": float(np.median(t)),
            "p99_ms":    float(np.percentile(t, 99)),
            "max_ms":    float(np.max(t)),
        }


if __name__ == "__main__":
    agent = DQNAgent(cfg_path="../config.json")
    print("Agent initialised.")
    print(f"  State dim:  {agent.state_dim}")
    print(f"  Num beams:  {agent.num_beams}")
    print(f"  Buffer cap: {agent.buffer.capacity:,}")
    print(f"  Epsilon:    {agent.epsilon}")

    # Simulate 200 steps
    prev_beam = -1
    for step in range(200):
        state = np.random.randn(16).astype(np.float32)
        action, inf_ms = agent.select_action(
            state, 100.0, 50.0)
        sinr_db = np.random.uniform(10.0, 30.0)
        beam_changed = (action != prev_beam and
                        prev_beam >= 0)
        reward = agent.compute_reward(
            sinr_db, beam_changed)
        next_state = np.random.randn(16)\
                               .astype(np.float32)
        agent.buffer.push(
            state, action, reward, next_state, False)
        prev_beam = action

    print(f"\nAfter 200 steps:")
    print(f"  Buffer size: {len(agent.buffer)}")

    # Train 10 steps
    losses = []
    for _ in range(10):
        loss = agent.train_step()
        if loss > 0:
            losses.append(loss)

    print(f"  Train steps: 10")
    print(f"  Mean loss:   {np.mean(losses):.5f}")

    # Latency report (G-16)
    lr = agent.latency_report()
    print(f"\nInference latency (200 calls):")
    print(f"  Mean:   {lr['mean_ms']:.3f} ms")
    print(f"  Median: {lr['median_ms']:.3f} ms")
    print(f"  P99:    {lr['p99_ms']:.3f} ms")
    print(f"  Max:    {lr['max_ms']:.3f} ms")

    # Save/load checkpoint
    Path("../checkpoints").mkdir(exist_ok=True)
    agent.save("../checkpoints/test_agent.pt")
    agent.load("../checkpoints/test_agent.pt")

    print("\nDQN agent test PASSED")
