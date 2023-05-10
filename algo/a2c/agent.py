import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from algo.base import Algorithm
from algo.models import CNNActorCritic
from algo.utils import float_tensor
from utils.logger import Logger
from utils.summary import EvaluationMetrics


class A2C(Algorithm):
    # NOTE: this implementation currently only works for bandit-like situation,
    #       i.e., where the next state is always set as a terminal state
    def __init__(self, args):
        super().__init__(args)
        # initialize logger
        self.logger = Logger('A2C', args)
        self.logger.log("Initialized A2C agent for environment: {}".format(
            args.env_id
        ))

        # initialize models
        img_shape = self.env.observation_space.shape
        self.model = CNNActorCritic(
            img_shape,
            action_space=self.env.action_space,
            device=args.device
        )
        self.model.to(args.device)

        # intialize optimizer
        self.optim = optim.Adam(
            self.model.parameters(),
            lr=1e-4,
        )

        # initialize statistics
        self.info = EvaluationMetrics([
            'Time/Item',
            'Loss/Critic',
            'Loss/Actor',
            'Values/Entropy',
            'Values/Reward',
            'Values/Value',
            'Scores/Return',
            'Scores/BnH',
            'Scores/Baseline',
            'Scores/Gain',
        ])
        self.acs_buf = deque(maxlen=self.args.maxlen)

    def train(self):
        st = time.time()
        self.step += 1

        self.model.train()
        self.env.train()

        # collect transition
        acs, log_probs, ents, vals = self.model(self.obs)
        self.acs_buf.append(acs.detach())
        self.info.update('Values/Entropy', ents.mean().item())
        self.info.update('Values/Value', vals.mean().item())

        obs, rews, dones, info = self.env.step(acs.detach().cpu().numpy())
        rews = float_tensor(rews, device=self.args.device).unsqueeze(1)
        self.info.update('Values/Reward', rews.mean().item())

        # evaluate scores
        total_return = np.mean([buf['return'] for buf in info])
        bnh = np.mean([buf['bnh'] for buf in info])
        baseline = np.mean([buf['baseline'] for buf in info])
        gain = total_return - max(bnh, baseline)
        self.info.update('Scores/Return', total_return)
        self.info.update('Scores/BnH', bnh)
        self.info.update('Scores/Baseline', baseline)
        self.info.update('Scores/Gain', gain)

        # critic loss
        advs = rews - vals
        loss_critic = advs.pow(2).mean()
        self.info.update('Loss/Critic', loss_critic.item())

        # actor loss
        loss_actor = (advs.detach() * -log_probs).mean()
        self.info.update('Loss/Actor', loss_actor.item())

        # update model parameters
        loss = loss_actor + self.args.val_coef * loss_critic
        loss -= self.args.ent_coef * ents.mean()
        self.optim.zero_grad()
        loss.backward()
        if self.args.max_grad is not None:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.args.max_grad
            )
        self.optim.step()

        # get next states
        self.obs = float_tensor(obs, device=self.args.device)

        # log training statistics
        elapsed = time.time() - st
        self.info.update('Time/Item', elapsed, n=self.args.n_env)
        if self.step % self.args.log_step == 0:
            self.logger.log(
                "Summary of statistics at step {}".format(self.step))
            self.logger.scalar_summary(self.info.avg, self.step)
            self.info.reset()
            self.plot_actions()

        if self.step % self.args.save_step == 0:
            self.save_model()
