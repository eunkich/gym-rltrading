from backtesting import Strategy
from backtesting.lib import crossover
import numpy as np

import pandas as pd
from gym import spaces


def MFI(high, low, close, volume, window_size):
    """
    Money Flow Index:
        Mixture of price + volume index. It's folmula is every much
        related to RSI index

        Works well in box pattern, but you may need other strategies
        when applying on trend market.

        The default window for MFI is 14 days. Overbought/Oversold are
        set in 80% and 20% respectively.
    """
    typical_price = (high + low + close) / 3

    signs = [1]
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            signs.append(1)
        elif typical_price[i] < typical_price[i-1]:
            signs.append(-1)
        else:
            signs.append(0)

    money_flow = volume * typical_price * signs
    pos_mf = money_flow
    neg_mf = money_flow.copy()

    pos_mf[pos_mf < 0] = 0
    neg_mf[neg_mf > 0] = 0

    money_flow_ratio = pd.Series(pos_mf).rolling(
        window_size).sum() / pd.Series(neg_mf).rolling(window_size).sum() * (-1)

    mfi = 100 * (1 - 1 / (1 + money_flow_ratio))
    #import pdb
    # pdb.set_trace()
    return mfi


class MFICross(Strategy):
    n1, n1_base = None, 2  # buying point
    n2, n2_base = None, 11  # selling point
    n3, n3_base = None, 17  # window size

    action_space = spaces.Box(
        np.array([0, 50, 2], dtype=np.float32),
        np.array([50, 100, 100], dtype=np.float32),
        dtype=np.float32
    )

    def init(self):
        self.n3 = int(round(self.n3))
        self.mfi = self.I(MFI, self.data.High, self.data.Low,
                          self.data.Close, self.data.Volume, self.n3)

    def next(self):
        if crossover(self.mfi, self.n1):
            self.buy()
        elif crossover(self.n2, self.mfi):
            self.sell()
