import d4rl_pybullet
import gym
import numpy as np
import torch
from torch.utils.data import Dataset

D4RL_ENV_NAMES = ['ant-bullet-medium-v0', 'halfcheetah-bullet-medium-v0', 'hopper-bullet-medium-v0', 'walker2d-bullet-medium-v0']


class D4RLEnv():
    def __init__(self, env_name):
        assert env_name in D4RL_ENV_NAMES

        self.env = gym.make(env_name)
        self.env.action_space.high, self.env.action_space.low = torch.as_tensor(
            self.env.action_space.high), torch.as_tensor(
            self.env.action_space.low)  # Convert action space for action clipping

    def reset(self):
        state = self.env.reset()
        return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state

    def step(self, action):
        action = action.clamp(min=self.env.action_space.low, max=self.env.action_space.high)  # Clip actions
        state, reward, terminal, _ = self.env.step(action[0].detach().numpy())  # Remove batch dimension from action
        return torch.tensor(state, dtype=torch.float32).unsqueeze(
            dim=0), reward, terminal  # Add batch dimension to state

    def seed(self, seed):
        return self.env.seed(seed)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def get_dataset(self, size=0, subsample=100):
        dataset = self.env.get_dataset()
        N = dataset['rewards'].shape[0]
        dataset_out = {'states': torch.as_tensor(dataset['observations'][:-1], dtype=torch.float32),
                       'actions': torch.as_tensor(dataset['actions'][:-1], dtype=torch.float32),
                       'rewards': torch.as_tensor(dataset['rewards'][:-1], dtype=torch.float32),
                       'next_states': torch.as_tensor(dataset['observations'][1:], dtype=torch.float32),
                       'terminals': torch.as_tensor(dataset['terminals'][:-1], dtype=torch.float32)}
        # Postprocess
        if size > 0 and size < N:
            for key in dataset_out.keys():
                dataset_out[key] = dataset_out[key][0:size]
        if subsample > 0:
            for key in dataset_out.keys():
                dataset_out[key] = dataset_out[key][0::subsample]  # take one tuple every subsample

        return TransitionDataset(dataset_out)


# Dataset that returns transition tuples of the form (s, a, r, s', terminal)
class TransitionDataset(Dataset):
  def __init__(self, transitions):
    super().__init__()
    self.states, self.actions, self.rewards, self.terminals = transitions['states'], transitions['actions'].detach(), transitions['rewards'], transitions['terminals']  # Detach actions

  # Allows string-based access for entire data of one type, or int-based access for single transition
  def __getitem__(self, idx):
    if isinstance(idx, str):
      if idx == 'states':
        return self.states
      elif idx == 'actions':
        return self.actions
      elif idx == 'terminals':
        return self.terminals
    else:
      return dict(states=self.states[idx], actions=self.actions[idx], rewards=self.rewards[idx], next_states=self.states[idx + 1], terminals=self.terminals[idx])

  def __len__(self):
    return self.terminals.size(0) - 1  # Need to return state and next state


ENVS = {'ant': D4RLEnv, 'halfcheetah': D4RLEnv, 'hopper': D4RLEnv, 'walker2d': D4RLEnv}

