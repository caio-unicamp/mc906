import numpy as np
import pandas as pd
from agents.q_learning import QLearningAgent 

class SARSAAgent(QLearningAgent):
    """
    SARSA agent (On-Policy TD Control) for StockMarketEnv.

    This class inherit from QLearningAgent to reuse initialization, epsilon decay, and action selection.
    """

    def update(self, state, action, reward, next_state, next_action, done=False):
        """
        SARSA update:
          Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
          
        this method differs a little from its father's cause it uses the next action for the Q_value update
        """
        a_idx = self.action_to_idx[int(action)]
        next_a_idx = self.action_to_idx[int(next_action)]

        current_q = self.q_table[state][a_idx]
        
        next_q = 0.0 if done else self.q_table[next_state][next_a_idx]
        
        target = reward + self.gamma * next_q

        self.q_table[state][a_idx] = current_q + self.alpha * (target - current_q)