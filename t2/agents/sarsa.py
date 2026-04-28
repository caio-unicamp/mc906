import numpy as np
import pandas as pd

class SARSAAgent:
    """
    SARSA agent (On-Policy TD Control) for StockMarketEnv.

    External action space expected by env:
      -1 = sell, 0 = hold, 1 = buy
    """

    def __init__(
        self,
        env,
        learning_rate=0.1,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay_type="constant",  # "constant" | "linear" | "exponential"
        epsilon_linear_decay_steps=3500,
        epsilon_exp_decay_rate=0.9991,
        seed=42,
    ):
        self.env = env
        self.alpha = learning_rate
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_type = epsilon_decay_type
        self.epsilon = 0.2 if self.epsilon_decay_type == "constant" else self.epsilon_start
        self.epsilon_linear_decay_steps = epsilon_linear_decay_steps
        self.epsilon_exp_decay_rate = epsilon_exp_decay_rate

        self.actions = np.array([-1, 0, 1], dtype=int)
        self.action_to_idx = {-1: 0, 0: 1, 1: 2}
        self.idx_to_action = {0: -1, 1: 0, 2: 1}

        self.rng = np.random.default_rng(seed)
        # O shape deve corresponder às dimensões do estado retornadas pelo env
        self.q_table = np.zeros((2, 2, 2, 2, 3))

    def choose_action(self, state, training=True):
        """
        training=True  -> epsilon-greedy
        training=False -> greedy (inference)
        """

        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(self.actions))

        a_idx = int(np.argmax(self.q_table[state]))
        return self.idx_to_action[a_idx]

    def update(self, state, action, reward, next_state, next_action, done=False):
        """
        SARSA update:
          Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        """
        a_idx = self.action_to_idx[int(action)]
        next_a_idx = self.action_to_idx[int(next_action)]

        current_q = self.q_table[state][a_idx]
        next_q = 0.0 if done else self.q_table[next_state][next_a_idx]
        
        target = reward + self.gamma * next_q

        self.q_table[state][a_idx] = current_q + self.alpha * (target - current_q)

    def update_epsilon(self, episode):
        episode = int(episode)

        if self.epsilon_decay_type == "constant":
            self.epsilon = self.epsilon
        elif self.epsilon_decay_type == "linear":
            frac = min(1.0, episode / max(1, self.epsilon_linear_decay_steps))
            self.epsilon = self.epsilon_start - frac * (self.epsilon_start - self.epsilon_min)
        elif self.epsilon_decay_type == "exponential":
            self.epsilon = self.epsilon * self.epsilon_exp_decay_rate

        self.epsilon = max(self.epsilon_min, self.epsilon)

    def q_table_as_dataframe(self):
        rows = []
        for idx in np.ndindex(self.q_table.shape[:-1]):
            q_vals = self.q_table[idx]
            rows.append(
                {
                    "state": idx,
                    "q_sell(-1)": q_vals[0],
                    "q_hold(0)": q_vals[1],
                    "q_buy(1)": q_vals[2],
                }
            )
        return pd.DataFrame(rows)