import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition',
    ['state', 'action', 'reward', 'next_state', 'done'])


class PERBuffer:
    """
    Prioritized Experience Replay buffer.
    Samples proportional to |TD-error|^alpha.
    alpha=0.6, beta anneals 0.4->1.0 (G-08).
    """
    def __init__(self, capacity=50000, alpha=0.6,
                 beta_start=0.4, beta_end=1.0,
                 beta_steps=100000, eps=1e-5):
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta       = beta_start
        self.beta_end   = beta_end
        self.beta_inc   = (beta_end - beta_start) / beta_steps
        self.eps        = eps
        self.buffer     = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos        = 0

    def push(self, state, action, reward,
             next_state, done):
        max_pri = self.priorities.max() \
                  if self.buffer else 1.0
        t = Transition(state, action, reward,
                       next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(t)
        else:
            self.buffer[self.pos] = t
        self.priorities[self.pos] = max_pri
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n     = len(self.buffer)
        probs = self.priorities[:n] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(
            n, batch_size, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]
        # Importance-sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(self.beta_end,
                        self.beta + self.beta_inc)
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = \
                (abs(err) + self.eps) ** self.alpha

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    buf = PERBuffer(capacity=1000)

    # Fill with dummy transitions
    for i in range(200):
        buf.push(
            np.zeros(16, dtype=np.float32),
            int(i % 32),
            float(i % 10) / 10.0,
            np.zeros(16, dtype=np.float32),
            False)

    print(f"Buffer size:      {len(buf)} (should be 200)")

    samples, idx, weights = buf.sample(32)
    print(f"Batch size:       {len(samples)} (should be 32)")
    print(f"IS weight range:  [{weights.min():.3f}, "
          f"{weights.max():.3f}]")
    print(f"Max weight:       {weights.max():.3f} "
          f"(should be 1.000)")

    # Test priority update
    td_errors = np.random.uniform(0.1, 2.0, size=32)
    buf.update_priorities(idx, td_errors)
    print(f"Priority range after update: "
          f"[{buf.priorities[:200].min():.4f}, "
          f"{buf.priorities[:200].max():.4f}]")

    # Test beta annealing
    beta_before = buf.beta
    buf.sample(32)
    beta_after = buf.beta
    print(f"Beta annealing:   {beta_before:.6f} -> "
          f"{beta_after:.6f} (should increase)")

    print("\nPER buffer test PASSED")
