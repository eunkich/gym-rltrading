from backtesting import Strategy
from backtesting.test import SMA
import numpy as np
from gym import spaces


def upperband(basic_band, close_price):
    out = np.zeros_like(basic_band)
    for i in range(1, len(out)):
        if np.isnan(basic_band[i]):
            continue
        if basic_band[i] < out[i - 1] or close_price[i - 1] > out[i - 1]:
            out[i] = basic_band[i]
        else:
            out[i] = out[i - 1]
    return out


def lowerband(basic_band, close_price):
    out = np.zeros_like(basic_band)
    for i in range(1, len(out)):
        if np.isnan(basic_band[i]):
            continue
        if basic_band[i] > out[i - 1] or close_price[i - 1] < out[i - 1]:
            out[i] = basic_band[i]
        else:
            out[i] = out[i - 1]
    return out


def supertrend(upper, lower, close_price):
    out = np.zeros_like(upper)
    for i in range(1, len(out)):
        if out[i - 1] == upper[i - 1]:
            out[i] = upper[i] if close_price[i] <= upper[i] else lower[i]
        else:
            out[i] = lower[i] if close_price[i] >= lower[i] else upper[i]
    return out


class Supertrend(Strategy):
    n1, n1_base = None, 7  # ATR range
    n2, n2_base = None, 3  # multiplier
    action_space = spaces.Box(
        np.array([1, 0], dtype=np.float32),
        np.array([24 * 18, 10], dtype=np.float32),
        dtype=np.float32
    )

    def init(self):
        self.n1 = round(self.n1)
        tr = np.stack([
            (self.data.High - self.data.Low)[1:],
            np.abs(self.data.High[1:] - self.data.Close[:-1]),
            np.abs(self.data.Low[1:] - self.data.Close[:-1]),
        ]).max(axis=0)
        tr = np.insert(tr, 0, np.nan)
        atr = SMA(tr, self.n1)
        mid = (self.data.High + self.data.Low) / 2
        close = self.data.Close
        self.upper = self.I(upperband, mid + self.n2 * atr, close)
        self.lower = self.I(lowerband, mid - self.n2 * atr, close)
        self.supertrend = self.I(supertrend, self.upper, self.lower, close)

    def next(self):
        prev = self.upper[-2] == self.supertrend[-2]
        curr = self.upper[-1] == self.supertrend[-1]
        if prev != curr:
            self.buy() if prev else self.sell()
