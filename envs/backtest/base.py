from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from functools import partial
from backtesting import Backtest

INIT_CASH = 1e9
COMMISSION = .002


class Candles:
    sampling_rate = "10min"
    train = 3 * 24 * 6      # 3 days
    val = 1 * 24 * 6        # 1 day
    test = 30 * 24 * 6      # 30 days


class BacktestSingle(ABC):
    def __init__(self, datapath=None):
        self.raw = pd.read_pickle(datapath)
        self.raw = self.raw.resample(Candles.sampling_rate).agg(
            {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
        )
        self.raw['Open'].fillna(method='ffill', inplace=True)
        self.raw['High'].fillna(method='ffill', inplace=True)
        self.raw['Low'].fillna(method='ffill', inplace=True)
        self.raw['Close'].fillna(method='ffill', inplace=True)
        self.raw['Volume'].fillna(0, inplace=True)

        self.bt = partial(Backtest, commission=COMMISSION, cash=INIT_CASH,
                          exclusive_orders=True)
        self.observation_space = None
        self.action_space = None
        self.training = True

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def render(self, mode='human'):
        return None

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
