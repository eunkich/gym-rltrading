import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class CNNEncoder(nn.Module):
    def __init__(self, img_shape, hidden=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_shape[0], 16, 3),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )
        size = self.features(torch.zeros(1, *img_shape)).view(-1).size(0)
        self.fc = nn.Sequential(
            nn.Linear(size, hidden),
            Mish(),
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out


class CNNActor(nn.Module):
    def __init__(self, img_shape, hidden=128, action_space=None, device=None):
        super().__init__()
        self.enc = CNNEncoder(img_shape, hidden=hidden)
        self.ac_head = DiagGaussianHead(
            hidden,
            action_space=action_space,
            device=device
        )

    def forward(self, obs):
        states = self.enc(obs)
        acs, log_probs, _ = self.ac_head(states)
        return acs, log_probs


class CNNCritic(nn.Module):
    def __init__(self, img_shape, nb_actions, hidden=128):
        super().__init__()
        self.enc = CNNEncoder(img_shape, hidden=hidden)
        self.val_head = nn.Sequential(
            nn.Linear(hidden + nb_actions, hidden),
            Mish(),
            nn.Linear(hidden, hidden),
            Mish(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs, acs):
        states = self.enc(obs)
        z = torch.cat([states, acs], dim=1)
        vals = self.val_head(z)
        return vals


class CNNActorCritic(nn.Module):
    def __init__(self, img_shape, hidden=128, action_space=None, device=None):
        super().__init__()
        self.enc = CNNEncoder(img_shape, hidden=hidden)
        self.ac_head = DiagGaussianHead(
            hidden,
            action_space=action_space,
            device=device
        )
        self.val_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        states = self.enc(obs)
        acs, log_probs, ents = self.ac_head(states)
        vals = self.val_head(states)
        return acs, log_probs, ents, vals


class DiagGaussianHead(nn.Module):
    def __init__(self, hidden, action_space=None, device=None):
        super().__init__()
        self.fc = nn.Linear(hidden, action_space.shape[0] * 2)
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2)
        if device is not None:
            self.action_scale = self.action_scale.to(device)
            self.action_bias = self.action_bias.to(device)

    def forward(self, x):
        # create normal distribution
        out = self.fc(x)
        mu = out[..., :out.size(-1) // 2]
        sig = torch.sqrt(F.softplus(out[..., out.size(-1) // 2:]) + 1e-5)
        dist = Normal(mu, sig)

        # random sampling
        ac_raw = dist.rsample() if self.training else mu
        ac_clip = torch.tanh(ac_raw)
        ac_out = ac_clip * self.action_scale + self.action_bias

        # log probability
        log_prob = dist.log_prob(ac_raw)
        log_prob -= torch.log(self.action_scale * (1 - ac_clip.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        # entropy
        entropy = dist.entropy()
        return ac_out, log_prob, entropy
