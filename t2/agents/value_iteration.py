import numpy as np


class ValueIterationAgent:

    def __init__(
        self,
        env,
        gamma=0.9,
        theta=1e-4,
        max_iterations=1000,
        transaction_cost_override=None,
        state_shape=None,
        actions=None,
    ):
        """
        Generalized Value Iteration Agent.
        
        Args:
            env: Environment with state indicators and log-returns
            gamma: Discount factor
            theta: Convergence threshold for Bellman backup
            max_iterations: Maximum number of sweeps
            transaction_cost_override: Custom transaction cost (default 0.01)
            state_shape: Tuple defining state space shape (default (2,2,2,2))
            actions: Array of possible actions (default [-1, 0, 1])
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.transaction_cost_override = transaction_cost_override if transaction_cost_override is not None else 0.01

        # Actions
        self.actions = np.array(actions if actions is not None else [-1, 0, 1], dtype=int)
        self.action_to_idx = {int(a): i for i, a in enumerate(self.actions)}
        self.idx_to_action = {i: int(a) for i, a in enumerate(self.actions)}

        # State shape (default: 4D state space for (position, vol_state, macd_state, rsi_state))
        self.state_shape = state_shape if state_shape is not None else (2, 2, 2, 2)
        
        self.value_table = np.zeros(self.state_shape)
        self.policy_table = np.zeros(self.state_shape, dtype=int)
        self.transition_model = self._build_transition_model()


    def _build_transition_model(self):
        """
        Build an empirical transition model from the current environment data.

        The environment is episodic and the observable state is discrete, so the
        transition model is estimated by counting how often each (state, action)
        pair leads to a next_state and reward across the historical trajectory.
        """
        vol_state = np.asarray(self.env._vol_state, dtype=int)
        macd_state = np.asarray(self.env._macd_state, dtype=int)
        rsi_state = np.asarray(self.env._rsi_state, dtype=int)
        log_returns = np.asarray(self.env._log_returns, dtype=float)

        episode_len = int(getattr(self.env, "episode_len", len(self.env.prices)))
        last_t = min(len(log_returns), episode_len - 1)

        model = {}

        for t in range(last_t):
            current_indicators = (
                int(vol_state[t]),
                int(macd_state[t]),
                int(rsi_state[t]),
            )

            if t + 1 < len(vol_state):
                next_indicators = (
                    int(vol_state[t + 1]),
                    int(macd_state[t + 1]),
                    int(rsi_state[t + 1]),
                )
            else:
                next_indicators = current_indicators

            price_move = float(log_returns[t])
            is_terminal = t >= last_t - 1

            for position in (0, 1):
                state = (position, *current_indicators)

                for action in self.actions:
                    next_position = position
                    transaction = False

                    if action == 1 and position == 0:
                        next_position = 1
                        transaction = True
                    elif action == -1 and position == 1:
                        next_position = 0
                        transaction = True

                    transaction_cost = self.transaction_cost_override if transaction else 0.0
                    reward = price_move * next_position - transaction_cost
                    next_state = (next_position, *next_indicators)

                    state_bucket = model.setdefault(state, {})
                    action_bucket = state_bucket.setdefault(int(action), {})
                    sample = action_bucket.setdefault(
                        next_state,
                        {"count": 0, "reward_sum": 0.0, "terminal_count": 0},
                    )
                    sample["count"] += 1
                    sample["reward_sum"] += reward
                    if is_terminal:
                        sample["terminal_count"] += 1

        transition_model = {}
        for state, action_dict in model.items():
            for action, next_state_dict in action_dict.items():
                total_count = sum(sample["count"] for sample in next_state_dict.values())
                if total_count == 0:
                    continue

                normalized = {}
                for next_state, sample in next_state_dict.items():
                    normalized[next_state] = {
                        "probability": sample["count"] / total_count,
                        "reward": sample["reward_sum"] / sample["count"],
                        "terminal": sample["terminal_count"] > 0,
                    }

                transition_model[(state, action)] = normalized

        return transition_model


    def refresh_transition_model(self):
        self.transition_model = self._build_transition_model()
        return self.transition_model


    def update(self, state, transition_model=None):
        """
        Bellman optimality backup:
          V(s) <- max_a sum_s' T(s, a, s') [R(s, a, s') + gamma V(s')]

        The transition_model must map (state, action) -> next-state outcomes.
        Each outcome can be either:
          - next_state -> (probability, reward)
          - next_state -> {"probability": p, "reward": r}
        """
        transition_model = transition_model or self.transition_model
        if transition_model is None:
            raise ValueError("A transition_model is required for Bellman updates.")

        state = tuple(state)

        best_value = -np.inf
        best_action = 0

        for action in self.actions:
            action_value = 0.0
            transitions = transition_model.get((state, int(action)), {})

            for next_state, transition in transitions.items():
                if isinstance(transition, dict):
                    probability = float(transition.get("probability", transition.get("p", 0.0)))
                    reward = float(transition.get("reward", 0.0))
                    terminal = bool(transition.get("terminal", False))
                else:
                    probability, reward = transition
                    terminal = False

                next_value = 0.0 if terminal else self.value_table[tuple(next_state)]
                action_value += probability * (reward + self.gamma * next_value)

            if action_value > best_value:
                best_value = action_value
                best_action = int(action)

        if best_value == -np.inf:
            best_value = 0.0

        self.value_table[state] = best_value
        self.policy_table[state] = best_action


    def choose_action(self, state, training=True):
        """
        training=True  -> epsilon-greedy
        training=False -> greedy (inference)
        """
        state = tuple(state)

        if training and np.random.random() < 0.1:
            return int(np.random.choice(self.actions))

        return int(self.policy_table[state])
