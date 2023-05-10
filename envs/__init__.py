import os
from gym.envs.registration import register
import envs.backtest as backtest
import strategies
from envs.trading import trading


# Iterate over data files
for filename in os.listdir('data'):
    buy, sell = filename.split('_')[:2]

    # Iterate over indicator strategies
    for strat_name in dir(strategies):
        strat = getattr(strategies, strat_name)
        if hasattr(strat, 'init'):
            register(
                id='{}/{}-{}-v0'.format(buy, sell, strat_name).lower(),
                entry_point='envs.trading:Trading',
                kwargs={
                    'backtest': getattr(backtest, 'BacktestIndicator'),
                    'datapath': 'data/' + filename,
                    'strategy': strat
                }
            )
