from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
from gym import spaces


def simplema(arr: np.array, n: int) -> np.array:
    """
    Returns `n`-period simple moving average of array `arr`.
    """
    out = pd.Series(arr).rolling(n).mean()
    out = np.array(out)
    return out

def upperbb(arr: np.array, sma: np.array, n: int, k: float):
    """
    Returns `n`-period, `k`-width upper bollinger band of array `arr`.
    """
    sd = pd.Series(arr).rolling(n).std()
    sd = np.array(sd)
    out = sma + (k * sd)
    
    return out
 
def lowerbb(arr: np.array, sma: np.array, n: int, k:float):
    """
    Returns `n`-period, `k`-width lower bollinger band of array `arr`.
    """
    sd = pd.Series(arr).rolling(n).std()
    sd = np.array(sd)
    out = sma - (k * sd)
    
    return out

def percentb(price: np.array, ubb: np.array, lbb:np.array):
    """
    Returns %B of array `price`; (price-lbb)/(ubb-lbb).
    """
    out = (price-lbb)/(ubb-lbb)
    out = pd.Series(out)
    
    return out

def bandwidth(price: np.array, ubb: np.array, lbb:np.array, n:int):
    """
    Returns bandwidth of array `price`; (price-lbb)/(ubb-lbb).
    """
    bw = pd.Series(ubb-lbb)
    avgbw = bw.rolling(n).mean()
    out = bw/avgbw
    return out
 

    

class BbCross(Strategy):
    n1, n1_base = None, 20 # BB window
    n2, n2_base = None, 2.0 # SD multiplier
    n3, n3_base = None, 0.6 # BW threshold

    action_space = spaces.Box(
        np.array([1, 0, 0], dtype=np.float32),
        np.array([500, 20, 1], dtype=np.float32),
        dtype=np.float32
    )

    def init(self):
        self.n1 = int(round(self.n1))
        self.n2 = round(self.n2, 4)
        self.n3 = round(self.n3, 4)
        self.price = (self.data.Close + self.data.High + self.data.Low)/3
        self.sma = self.I(simplema, self.price, n=self.n1)
        self.ubb = self.I(upperbb, self.price, self.sma, n=self.n1, k=self.n2)
        self.lbb = self.I(lowerbb, self.price, self.sma, n=self.n1, k=self.n2)
        self.pb = self.I(percentb, self.price, self.ubb, self.lbb) 
        self.bw = self.I(bandwidth, self.price, self.ubb, self.lbb, n=self.n1)       


    def next(self):
        # if self.pb[-1] > 1 and self.bw[-1]<self.n3:
        #     self.buy()
        # elif self.pb[-1] < 0 and self.bw[-1]<self.n3:
        #     self.sell()
        
        # Inv
        if self.pb[-1] > 1 and self.bw[-1]<self.n3:
            self.sell()
        elif self.pb[-1] < 0 and self.bw[-1]<self.n3:
            self.buy()