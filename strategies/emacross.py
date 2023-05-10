from backtesting import Strategy
from backtesting.lib import crossover
import numpy as np
from pandas import Series
from gym import spaces


def EMA(arr, n, k):
    weights=np.power(1-k, list(reversed(range(n))))
    weights/=np.sum(weights)
    return Series(arr).rolling(n).apply(lambda x: np.sum(weights*x), raw=False)
    
class EmaCross(Strategy):
    n1, n1_base = None, 13  # short EMA
    n2, n2_base = None, 48  # long EMA
    k = 0.1
    action_space = spaces.Box(
        np.array([1, 1], dtype=np.float32),
        np.array([24 * 18, 24 * 18], dtype=np.float32),
        dtype=np.float32
    )

    def init(self):
        self.n1 = int(round(self.n1))
        self.n2 = int(round(self.n2))
        self.ema1 = self.I(EMA, self.data.Close, self.n1, self.k)
        self.ema2 = self.I(EMA, self.data.Close, self.n2, self.k)

    def next(self):
        if crossover(self.ema1, self.ema2):
            self.buy()
        elif crossover(self.ema2, self.ema1):
            self.sell()

