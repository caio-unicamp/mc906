import numpy as np
import pandas as pd

class StockMarketEnv():
    """
    A simple stock market environment for reinforcement learning.
    Position: 0 = no position, 1 = long
    Actions: 1 = Buy, -1 = Sell, 0 = Hold
    """
    def __init__(self, data_path, window_size=10, noise_pct=0.05):
        # Lê dataset e guarda cópia original dos preços
        self.original_prices = pd.read_csv(data_path)
        self.noise_pct = noise_pct

        # Primeira inicialização (sem ruído)
        self.prices = self.original_prices.copy()
        self._calculate_indicators()
        self.episode_len = len(self.prices)

        self.window_size = window_size
        self.position = 0
        self.current_step = 0
        self.actions = np.array([-1, 0, 1], dtype=int)
        

    def step(self, action): 
        # action: 0 = hold, 1 = buy, -1 = sell
        # r = log(P_t / P_{t-1}) * position        
        self.current_step += 1

        transaction = False
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            transaction = True
        elif action == -1 and self.position == 1:  # Sell
            self.position = 0
            transaction = True
        transaction_cost = 0.01 if transaction else 0

        reward = self._log_returns[self.current_step - 1] * self.position - transaction_cost
        done = self.current_step >= self.episode_len - 1
        state = self._get_state()

        return state, reward, done

    def _get_state(self):
        """
        Returns discretized state as tuple:
        (position, trend_hold, volatility_state, macd_state, rsi_state)
        Each value is 1 (holding) or -1 (not holding)
        """
        if self.current_step < 26:
            return (self.position, 0, 0, 0)

        return (self.position,
                self._vol_state[self.current_step],
                self._macd_state[self.current_step],
                self._rsi_state[self.current_step])

    def reset(self):
        # Aplica ruído multiplicativo nos preços a cada episódio
        if self.noise_pct > 0:
            noise = np.random.uniform(
                1 - self.noise_pct, 1 + self.noise_pct,
                size=len(self.original_prices),
            )
            self.prices = self.original_prices.copy()
            for col in ["Open", "High", "Low", "Close"]:
                self.prices[col] = self.original_prices[col] * noise
            self._calculate_indicators()

        self.current_step = 0
        self.position = 0
        return self._get_state()

    def render(self):
        current_price = self.prices["Close"].iloc[self.current_step]
        print(f"Step: {self.current_step}, Position: {self.position}, Close: {current_price:.2f}")

    def _calculate_indicators(self):
        """Calculate MACD, RSI and Volatility indicators for the whole data_set."""
        # Calculate MACD
        exp1 = self.prices["Close"].ewm(span=12, adjust=False).mean() #Exponential Moving Average with span of 12
        exp2 = self.prices["Close"].ewm(span=26, adjust=False).mean() #Exponential Moving Average with span of 26
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        self.prices["MACD"] = macd - signal
        self.prices["MACD_State"] = np.where(macd > signal, 1, 0) # 1 if MACD above signal, else 0

        # Calculate RSI
        delta = self.prices["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.prices["RSI"] = 100 - (100 / (1 + rs))
        self.prices["RSI_State"] = np.where(self.prices["RSI"] > 70, 1, 0) # 1 if RSI above 70, else 0
        
        # Calculate Volatility
        self.prices["Volatility"] = self.prices["Close"].pct_change().rolling(window=10).std()
        self.prices["Volatility_State"] = np.where(self.prices["Volatility"] < 0.015, 1, 0)

        # Cache columns as NumPy arrays for fast indexing in step()/get_state()
        self._close = self.prices["Close"].values
        self._vol_state = self.prices["Volatility_State"].values.astype(int)
        self._macd_state = self.prices["MACD_State"].values.astype(int)
        self._rsi_state = self.prices["RSI_State"].values.astype(int)
        self._log_returns = np.diff(np.log(self._close))