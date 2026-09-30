"""Checkpointable on-policy PPO, with raw Gaussian actions in likelihood ratios.

The reference profile uses the released FLRL CNN shape and PPO hyperparameters.
This worker-driven PyTorch implementation replaces SB3's synchronous Env loop;
it is an adaptation, not a claim of bitwise SB3 or paper-result reproduction.
"""
from copy import deepcopy

import numpy as np
import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, observation_size, action_size, *, frames=1, cnn=False):
        super().__init__()
        self.frames, self.cnn = frames, cnn
        self.features = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()) if cnn else nn.Identity()
        width = 128 if cnn else observation_size
        def branch(output):
            return nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, output))
        self.actor, self.critic = branch(action_size), branch(1)
        self.log_std = nn.Parameter(torch.zeros(action_size))
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, np.sqrt(2)); nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, .01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.)

    def forward(self, observations):
        if self.cnn:
            observations = observations.reshape(len(observations), 1, -1, self.frames)
        features = self.features(observations)
        return torch.distributions.Normal(self.actor(features), self.log_std.clamp(-10, 2).exp()), self.critic(features).flatten()


class PPO:
    def __init__(self, observation_size, action_size, seed, config, *, cnn=False, frames=1):
        self.config = dict(config)
        self.rng = np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = ActorCritic(observation_size, action_size, cnn=cnn, frames=frames)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.get('ppo_learning_rate', .003), eps=1e-5)
        self.rollout, self.updates = [], 0
        self.gamma, self.lam = config.get('ppo_gamma', .9), config.get('ppo_gae_lambda', .95)
        self.return_value = 0.
        self.reward_count, self.reward_mean, self.reward_m2 = 1e-4, 0., 1e-4
        self.last_loss = 0.

    def sample(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        with torch.no_grad():
            distribution, value = self.network(torch.tensor(obs[None]))
            action = distribution.mean[0].numpy() + distribution.stddev[0].numpy() * self.rng.normal(size=distribution.mean.shape[-1])
            logp = distribution.log_prob(torch.tensor(action[None], dtype=torch.float32)).sum(-1).item()
        return {"observation": obs, "action": action.astype(np.float32), "logp": logp, "value": value.item(),
                "gaussian_mean": distribution.mean[0].numpy().copy(), "gaussian_std": distribution.stddev[0].numpy().copy()}

    def record(self, sample, reward, next_observation, done, *, normalize_reward=False, time_limit=False):
        if normalize_reward:
            self.return_value = reward + self.gamma * self.return_value
            self.reward_count += 1
            delta = self.return_value - self.reward_mean
            self.reward_mean += delta / self.reward_count
            self.reward_m2 += delta * (self.return_value - self.reward_mean)
            reward = float(np.clip(reward / np.sqrt(self.reward_m2 / self.reward_count + 1e-8), -10, 10))
            if done: self.return_value = 0.
        if done and time_limit:
            # The released reference wraps episodes in Gym TimeLimit. Bootstrap
            # its terminal observation, while GAE still stops at the reset.
            with torch.no_grad():
                _, terminal_value = self.network(torch.tensor(np.asarray(next_observation, dtype=np.float32)[None]))
            reward += self.gamma * terminal_value.item()
        self.rollout.append({**sample, "reward": float(reward), "done": done})
        length = self.config.get('rollout_steps', self.config.get('episode_length', 128))
        # The hybrid's frozen local kernel is trained only after whole episodes.
        if len(self.rollout) < length or self.config.get('complete_rollouts', False) and not done:
            return
        with torch.no_grad():
            _, value = self.network(torch.tensor(np.asarray(next_observation, dtype=np.float32)[None]))
        advantage, advantages = 0., []
        next_value = 0. if done else value.item()
        for row in reversed(self.rollout):
            continuation = 1. - float(row['done'])
            delta = row['reward'] + self.gamma * next_value * continuation - row['value']
            advantage = delta + self.gamma * self.lam * continuation * advantage
            advantages.append(advantage)
            next_value = row['value']
        advantages = np.array(advantages[::-1], dtype=np.float32)
        returns = torch.tensor(advantages + np.array([row['value'] for row in self.rollout], dtype=np.float32))
        adv = torch.tensor((advantages - advantages.mean()) / (advantages.std() + 1e-8))
        observations = torch.tensor(np.stack([row['observation'] for row in self.rollout]))
        actions = torch.tensor(np.stack([row['action'] for row in self.rollout]))
        old_logp = torch.tensor([row['logp'] for row in self.rollout])
        n = len(self.rollout)
        batch = self.config.get('batch_size', max(2, n // 2))
        for _ in range(self.config.get('ppo_n_epochs', 3)):
            permutation = self.rng.permutation(n)
            for start in range(0, n, batch):
                ix = permutation[start:start + batch]
                distribution, values = self.network(observations[ix])
                ratio = (distribution.log_prob(actions[ix]).sum(-1) - old_logp[ix]).exp()
                clip = self.config.get('ppo_clip_range', .4)
                loss = -torch.minimum(ratio * adv[ix], ratio.clamp(1 - clip, 1 + clip) * adv[ix]).mean()
                loss = loss + .1 * ((values - returns[ix]) ** 2).mean() - .01 * distribution.entropy().sum(-1).mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite PPO loss')
                self.optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), .5)
                self.optimizer.step()
                self.last_loss = float(loss.detach())
        self.updates += 1
        self.rollout = []

    def state_dict(self):
        return deepcopy({"network": self.network.state_dict(), "optimizer": self.optimizer.state_dict(),
            "rng": self.rng.bit_generator.state, **{k: getattr(self, k) for k in
                ('rollout', 'updates', 'return_value', 'reward_count', 'reward_mean', 'reward_m2', 'last_loss')}})

    def load_state_dict(self, state):
        self.network.load_state_dict(state['network']); self.optimizer.load_state_dict(state['optimizer'])
        self.rng.bit_generator.state = deepcopy(state['rng'])
        for key in ('rollout', 'updates', 'return_value', 'reward_count', 'reward_mean', 'reward_m2', 'last_loss'):
            setattr(self, key, deepcopy(state[key]))
