from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import numpy as np
from gym import spaces


class SmaCross(Strategy):
    n1, n1_base = None, 13  # short MA
    n2, n2_base = None, 48  # long MA
    action_space = spaces.Box(
        np.array([1, 1], dtype=np.float32),
        np.array([24 * 18, 24 * 18], dtype=np.float32),
        dtype=np.float32
    )

    def init(self):
        self.n1 = int(round(self.n1))
        self.n2 = int(round(self.n2))
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()
