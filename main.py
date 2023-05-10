import argparse
import random
import numpy as np
import torch
import logging
import os

# disable warning logs from the baselines module
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def train(args):
    import algo

    alg = getattr(algo, args.algo)(args)
    if args.checkpoint is not None:
        alg.logger.log("Loading model from {}".format(args.checkpoint))
        alg.load_model(args.checkpoint)
    for i in range(args.total_step):
        alg.train()


def optimize(args):
    from utils.logger import Logger
    from envs import trading
    from envs.backtest.base import BacktestSingle, Candles
    from skopt import Optimizer
    from skopt.space import Real
    from joblib import Parallel, delayed

    logger = Logger('skopt', args)

    # load single environment
    n_jobs = args.n_env
    args.n_env = 1
    env = trading(args).envs[0]
    assert hasattr(env, 'bt')
    env = env.bt
    assert isinstance(env, BacktestSingle)
    logger.log("Search optimal parameters for environment: " + args.env_id)
    data = env.raw[-(Candles.train + Candles.test):]  # optimal parameters for the test period
    bt = env.bt(data=data)

    # define optimizer
    optimizer = Optimizer(
        dimensions=[
            Real(env.action_space.low[i], env.action_space.high[i])
            for i in range(env.action_space.shape[0])
        ],
        random_state=args.seed,
        base_estimator='gp'
    )

    def objective_fn(v):
        actions = {'n' + str(i + 1): v[i] for i in range(len(v))}
        result = bt.run(**actions)
        return result["Return [%]"]

    for step in range(args.total_step):
        x = optimizer.ask(n_points=n_jobs)
        y = Parallel(n_jobs=n_jobs)(delayed(objective_fn)(v) for v in x)
        optimizer.tell(x, y)
        if (step + 1) % args.log_step == 0:
            msg = "Best configuration at step {}\nReturn: {}".format(
                step + 1, max(optimizer.yi)
            )
            idx = np.argmax(optimizer.yi)
            for i in range(len(x[0])):
                msg += "\nn{}: {}".format(str(i + 1), optimizer.Xi[idx][i])
            logger.log(msg)


def evaluate(args):
    import algo

    alg = getattr(algo, args.algo)(args)
    if args.checkpoint is not None:
        alg.logger.log("Loading model from {}".format(args.checkpoint))
        alg.load_model(args.checkpoint)
    total_return, bnh, baseline = alg.eval()
    alg.logger.log("Total Return (%): {}".format(total_return * 100))
    alg.logger.log("Buy & Hold Return (%): {}".format(bnh * 100))
    alg.logger.log("Baseline Return (%): {}".format(baseline * 100))


def test(args):
    from utils.logger import Logger

    logger = Logger('test', args)
    logger.log("Logger is working properly.")
    logger.log("Log files are saved in {}".format(logger.log_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Automated cryptocurrency trading using RL"
    )
    common = parser.add_argument_group("common configurations")
    common.add_argument("--mode", type=str, default='test')
    common.add_argument("--tag", type=str, default='')
    common.add_argument("--seed", type=int, default=-1)

    log = parser.add_argument_group("logging options")
    log.add_argument("--log_level", type=int, default=20)
    log.add_argument("--log_step", type=int, default=100)
    log.add_argument("--save_step", type=int, default=100)
    log.add_argument("--debug", "-d", action="store_true")
    log.add_argument("--quiet", "-q", action="store_true")
    log.add_argument("--maxlen", type=int, default=100)

    dirs = parser.add_argument_group("directory configurations")
    dirs.add_argument("--log_dir", type=str, default='logs')
    dirs.add_argument("--data_dir", type=str, default='data')
    dirs.add_argument("--checkpoint", type=str, default=None)

    training = parser.add_argument_group("training options")
    training.add_argument("--total_step", type=int, default=10000)
    training.add_argument("--env_id", type=str, default='btc/usdt-smacross-v0')
    training.add_argument("--n_env", type=int, default=1)
    training.add_argument("--algo", type=str, default='sac')
    training.add_argument("--val_coef", type=float, default=0.5)
    training.add_argument("--ent_coef", type=float, default=1e-3)
    training.add_argument("--max_grad", type=float, default=None)

    args = parser.parse_args()
    # set random seed
    if args.seed == -1:
        random.seed(None)
        args.seed = random.randrange(0, int(1e4))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # use cuda when available
    if not hasattr(args, 'device'):
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set logging level
    if args.debug:
        args.log_level = 1
    elif args.quiet:
        args.log_level = 30

    globals()[args.mode](args)
