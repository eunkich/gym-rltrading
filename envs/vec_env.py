import numpy as np
from torch.multiprocessing import Pipe, Process
import gym
from baselines.common import set_global_seeds
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
from baselines.common.vec_env.util import copy_obs_dict
from baselines.common.vec_env.util import dict_to_obs
from baselines.common.vec_env.util import obs_space_info
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'train':
                env.train()
            elif cmd == 'eval':
                env.eval()
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        self.specs = [f().spec for f in env_fns]
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return _flatten_obs([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def render(self, mode='human'):
        img = self.get_images()[0]
        if mode == 'human':
            self.get_viewer().imshow(img)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return img
        else:
            raise NotImplementedError

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def train(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('train', None))

    def eval(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('eval', None))


def _flatten_obs(obs):
    assert isinstance(obs, list) or isinstance(obs, tuple)
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        import collections
        assert isinstance(obs, collections.OrderedDict)
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)


class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns):
        """
        Arguments:
        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = {k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys}
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.specs = [e.spec for e in self.envs]

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            if isinstance(self.envs[e].action_space, gym.spaces.Discrete):
                action = int(action)

            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)
            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def train(self):
        for e in range(self.num_envs):
            self.envs[e].train()

    def eval(self):
        for e in range(self.num_envs):
            self.envs[e].eval()


def make_env(env_id, n_env, seed):
    def _make_env(env_id, subrank=0, seed=None):
        env = gym.make(env_id)
        if isinstance(env.observation_space, gym.spaces.Dict):
            keys = env.observation_space.spaces.keys()
            env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))
        env.seed(seed + subrank if seed is not None else None)
        return env

    def make_thunk(rank):
        return lambda: _make_env(
            env_id=env_id,
            subrank=rank,
            seed=seed
        )

    set_global_seeds(seed)
    if n_env > 1:
        return SubprocVecEnv([make_thunk(i) for i in range(n_env)])
    else:
        return DummyVecEnv([make_thunk(0)])
