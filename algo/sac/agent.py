import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from algo.base import Algorithm
from algo.models import CNNActor, CNNCritic
from algo.utils import float_tensor
from utils.logger import Logger
from utils.summary import EvaluationMetrics


class SAC(Algorithm):
    # NOTE: this implementation currently only works for bandit-like situation,
    #       i.e., where the next state is always set as a terminal state
    def __init__(self, args):
        super().__init__(args)
        # initialize logger
        self.logger = Logger('SAC', args)
        self.logger.log("Initialized SAC agent for environment: {}".format(
            args.env_id
        ))

        # initialize models
        img_shape = self.env.observation_space.shape
        nb_actions = self.env.action_space.shape[0]
        self.model = nn.ModuleDict({
            'actor': CNNActor(
                img_shape,
                action_space=self.env.action_space,
                device=args.device
            ),
            'critic': CNNCritic(img_shape, nb_actions),
        })
        self.model.to(args.device)

        # initialize entropy parameter
        self.target_alpha = -nb_actions  # -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        self.alpha = 1.0

        # intialize optimizers
        self.actor_optim = optim.Adam(
            self.model.actor.parameters(),
            lr=1e-4
        )
        self.critic_optim = optim.Adam(
            self.model.critic.parameters(),
            lr=1e-4
        )
        self.alpha_optim = optim.Adam(
            [self.log_alpha],
            lr=1e-4
        )

        # initialize statistics
        self.info = EvaluationMetrics([
            'Time/Item',
            'Loss/Critic',
            'Loss/Actor',
            'Loss/Alpha',
            'Values/Alpha',
            'Values/Reward',
            'Values/QValue',
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
        acs, log_probs = self.model.actor(self.obs)
        self.acs_buf.append(acs.detach())

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

        # update critic
        vals = self.model.critic(self.obs, acs.detach())
        self.info.update('Values/QValue', vals.mean().item())
        loss_critic = (rews - vals).pow(2).mean()
        self.info.update('Loss/Critic', loss_critic.item())
        self.critic_optim.zero_grad()
        loss_critic.backward()
        if self.args.max_grad is not None:
            nn.utils.clip_grad_norm_(
                self.model.critic.parameters(),
                self.args.max_grad
            )
        self.critic_optim.step()

        # update actor
        vals = self.model.critic(self.obs, acs)
        loss_actor = (self.alpha * log_probs - vals).mean()
        self.info.update('Loss/Actor', loss_actor.item())
        self.actor_optim.zero_grad()
        loss_actor.backward()
        if self.args.max_grad is not None:
            nn.utils.clip_grad_norm_(
                self.model.actor.parameters(),
                self.args.max_grad
            )
        self.actor_optim.step()

        # update entropy parameter
        loss_alpha = log_probs.detach() + self.target_alpha
        loss_alpha = (-self.log_alpha * loss_alpha).mean()
        self.info.update('Loss/Alpha', loss_alpha.item())
        self.alpha_optim.zero_grad()
        loss_alpha.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        self.info.update('Values/Alpha', self.alpha.item())

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

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        self.env.eval()

        obs = self.env.reset()
        obs = float_tensor(obs, device=self.args.device)
        done = False
        while not done:
            acs, _ = self.model.actor(obs)
            obs, _, dones, info = self.env.step(acs.cpu().numpy())
            obs = float_tensor(obs, device=self.args.device)
            done = any(dones)

        total_return = np.mean([buf['return'] for buf in info])
        bnh = np.mean([buf['bnh'] for buf in info])
        baseline = np.mean([buf['baseline'] for buf in info])
        return total_return, bnh, baseline
