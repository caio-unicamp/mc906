import numpy as np

class TradingEnv:
    def __init__(self, prices, window_size=10):
        self.prices = prices
        self.current_step = 0
        self.window_size = window_size

    def step(self, action): 
        # action: 0 = hold, 1 = buy, -1 = sell
        self.current_step += 1

        reward = np.log(self.prices[self.current_step] / self.prices[self.current_step - 1]) * action
        done = self.current_step >= len(self.prices) - 1
        
        state = self._get_state()

        return state, reward, done
    
    def _get_state(self):
        """
        Returns discretized state as tuple: (trend_hold, volatility_hold, momentum_hold, position_hold)
        Each value is 1 (holding) or -1 (not holding)
        """
        if self.current_step < self.window_size:
            return (-1, -1, -1, -1)
        
        window_prices = self.prices[self.current_step - self.window_size : self.current_step + 1]
        
        # 1. Trend: average return over window
        returns = np.diff(window_prices) / window_prices[:-1]
        trend = np.mean(returns)
        trend_hold = 1 if trend > 0.005 else -1
        
        # 2. Volatility: std of returns
        volatility = np.std(returns)
        volatility_hold = 1 if volatility < 0.05 else -1
        
        # 3. Momentum: recent vs average
        if len(returns) >= 2:
            momentum = returns[-1] - np.mean(returns[:-1])
        else:
            momentum = 0
        momentum_hold = 1 if momentum > 0.005 else -1
        
        # 4. Price position: current vs recent range
        min_price = np.min(window_prices)
        max_price = np.max(window_prices)
        if max_price > min_price:
            position = (self.prices[self.current_step] - min_price) / (max_price - min_price)
            position_hold = 1 if position > 0.5 else -1
        else:
            position_hold = -1
        
        return (trend_hold, volatility_hold, momentum_hold, position_hold)

    def reset(self):
        self.current_step = 0
        return self._get_state()
        