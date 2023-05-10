import gym
from envs.vec_env import make_env


class Trading(gym.Env):
    def __init__(self, backtest=None, **kwargs):
        self.bt = backtest(**kwargs)
        self.observation_space = self.bt.observation_space
        self.action_space = self.bt.action_space

    def reset(self):
        obs = self.bt.reset()
        return obs

    def step(self, action):
        obs, rew, done, info = self.bt.step(action)
        return obs, rew, done, info

    def render(self, mode='human'):
        self.bt.render(mode=mode)

    def seed(self, seed=None):
        self.bt.seed(seed)

    def train(self):
        self.bt.train()

    def eval(self):
        self.bt.eval()


def trading(args):
    return make_env(args.env_id, args.n_env, args.seed)
