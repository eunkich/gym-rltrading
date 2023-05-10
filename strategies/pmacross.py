from backtesting import Strategy
from backtesting.lib import crossover
import numpy as np
from pandas import Series
from gym import spaces


def PMA(arr, n, k):
    weights=np.power(list(reversed(range(1, n+1))), -k)
    weights/=np.sum(weights)
    return Series(arr).rolling(n).apply(lambda x: np.sum(weights*x), raw=False)
    
class PmaCross(Strategy):
    n1, n1_base = None, 13  # short PMA
    n2, n2_base = None, 48  # long PMA
    k = 0.3
    action_space = spaces.Box(
        np.array([1, 1], dtype=np.float32),
        np.array([24 * 18, 24 * 18], dtype=np.float32),
        dtype=np.float32
    )

    def init(self):
        self.n1 = int(round(self.n1))
        self.n2 = int(round(self.n2))
        self.pma1 = self.I(PMA, self.data.Close, self.n1, self.k)
        self.pma2 = self.I(PMA, self.data.Close, self.n2, self.k)

    def next(self):
        if crossover(self.pma1, self.pma2):
            self.buy()
        elif crossover(self.pma2, self.pma1):
            self.sell()

