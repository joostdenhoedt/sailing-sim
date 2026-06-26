"""
Q-table agent with epsilon-greedy exploration.

The Bellman update at the heart of this:

    Q(s, a)  ←  Q(s, a)  +  α [ r  +  γ · max_a' Q(s', a')  −  Q(s, a) ]
                                └──────── TD target ────────┘   └ current estimate ┘

- α  (alpha)   : learning rate  — how much we shift toward the new estimate
- γ  (gamma)   : discount       — how much we value future rewards vs immediate
- ε  (epsilon) : exploration    — probability of taking a random action instead of greedy

The table Q has shape (N_STATES, N_ACTIONS). Each cell is the agent's current
estimate of "total future reward if I'm in state s and take action a, then act
optimally afterwards."
"""

import numpy as np
from env import N_STATES, ACTION_CONTROLS, ACTION_NAMES, STATE_NAMES


class QAgent:
    def __init__(
        self,
        n_states:      int   = N_STATES,
        n_actions:     int   = len(ACTION_CONTROLS),
        alpha:         float = 0.15,   # learning rate
        gamma:         float = 0.95,   # discount factor
        epsilon:       float = 1.0,    # starting exploration rate
        epsilon_min:   float = 0.05,   # floor — always keep 5% exploration
        epsilon_decay: float = 0.997,  # multiply epsilon by this each episode
    ):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # The Q-table: initialised to zero (optimistic init could also work)
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)

        # Track how many times each state has been visited (useful for debugging)
        self.visit_counts = np.zeros(n_states, dtype=np.int32)

    # ── action selection ──────────────────────────────────────────────────────

    def act(self, state: int) -> int:
        """
        Epsilon-greedy: explore randomly with probability ε,
        otherwise exploit the best known action.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q[state]))

    # ── learning update ───────────────────────────────────────────────────────

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        """
        One Bellman backup.

        If done (terminal state), there is no future — target is just r.
        Otherwise the target is r + γ · best Q-value in the next state.
        """
        self.visit_counts[s] += 1

        if done:
            td_target = r
        else:
            td_target = r + self.gamma * np.max(self.q[s_next])

        td_error = td_target - self.q[s, a]   # how wrong was our estimate?
        self.q[s, a] += self.alpha * td_error  # nudge toward the truth

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── inspection / debugging ────────────────────────────────────────────────

    def print_qtable(self, top_n: int = 20):
        """
        Print the Q-table rows sorted by total Q-value (most learned first).
        Shows which states the agent cares about most.
        """
        row_max = self.q.max(axis=1)
        top_idx = np.argsort(row_max)[::-1][:top_n]

        header = f"{'State':<32} | " + " | ".join(f"{a:>10}" for a in ACTION_NAMES)
        print("\n" + "─" * len(header))
        print(header)
        print("─" * len(header))

        for s in top_idx:
            if row_max[s] < 0.01 and self.visit_counts[s] == 0:
                continue   # skip unvisited states
            vals    = " | ".join(f"{v:>10.3f}" for v in self.q[s])
            best    = ACTION_NAMES[int(np.argmax(self.q[s]))]
            visits  = self.visit_counts[s]
            print(f"{STATE_NAMES[s]:<32} | {vals}  ← {best}  (n={visits})")

        print("─" * len(header))

    def policy_summary(self):
        """
        One-line summary of the greedy policy for each TWA zone.
        Great for seeing at a glance what the agent has learned.
        """
        from env import N_SPEED, N_TRIM, N_TACK, TWA_ZONE_NAMES
        print("\nGreedy policy by TWA zone (most-visited speed/trim combo):")
        for tz in range(5):
            # Find the most visited state in this TWA zone
            base = tz * N_SPEED * N_TRIM * N_TACK
            chunk = self.q[base: base + N_SPEED * N_TRIM * N_TACK]
            visits = self.visit_counts[base: base + N_SPEED * N_TRIM * N_TACK]
            if visits.sum() == 0:
                print(f"  {TWA_ZONE_NAMES[tz]:<10}: not yet visited")
                continue
            best_local = int(np.argmax(chunk.max(axis=1)))
            best_action = ACTION_NAMES[int(np.argmax(chunk[best_local]))]
            print(f"  {TWA_ZONE_NAMES[tz]:<10}: {best_action}  "
                  f"(Q={chunk[best_local].max():.3f})")
