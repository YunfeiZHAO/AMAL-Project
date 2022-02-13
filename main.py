from collections import deque
import time

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from torch import optim
from tqdm import tqdm

from environments import ENVS, TransitionDataset

from models import ActorCritic, GAILDiscriminator, REDDiscriminator
from train import adversarial_imitation_update, ppo_update, target_estimation_update

from torch.utils.tensorboard import SummaryWriter


def flatten_list_dicts(list_dicts):
    return {k: torch.cat([d[k] for d in list_dicts], dim=0) for k in list_dicts[-1].keys()}


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    # Configuration check
    assert cfg.env_type in ENVS.keys()
    assert cfg.algorithm in ['GAIL', 'RED']
    # General setup
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Tensorboard initialisation
    subsample = 100
    outdir = str(subsample) + "/" +cfg.env_type + "/" + cfg.algorithm
    print(outdir)
    writer = SummaryWriter(outdir)

    # Set up environment
    env = ENVS[cfg.env_type](cfg.env_name)
    env.seed(cfg.seed)
    expert_trajectories = env.get_dataset()  # Load expert trajectories dataset
    state_size, action_size = env.observation_space.shape[0], env.action_space.shape[0]

    # Set up agent
    agent = ActorCritic(state_size, action_size, cfg.model.hidden_size, log_std_init=cfg.model.log_std_dev_init)
    agent_optimiser = optim.Adam(agent.parameters(), lr=cfg.reinforcement.learning_rate)
    # Set up imitation learning components

    if cfg.algorithm == 'GAIL':
        discriminator = GAILDiscriminator(state_size, action_size, cfg.model.hidden_size,
                                          state_only=cfg.imitation.state_only)
    elif cfg.algorithm == 'RED':
        discriminator = REDDiscriminator(state_size, action_size, cfg.model.hidden_size,
                                         state_only=cfg.imitation.state_only)
    discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=cfg.imitation.learning_rate)

    # Metrics
    metrics = dict()
    recent_returns = deque(maxlen=cfg.evaluation.average_window)  # Stores most recent evaluation returns

    # Main training loop
    state, terminal, intrinsic_reture, extrinsic_return, trajectories = env.reset(), False, torch.tensor([0.0]), 0, []
    if cfg.algorithm == 'GAIL':
        policy_trajectory_replay_buffer = deque(maxlen=cfg.imitation.replay_size)
    pbar = tqdm(range(1, cfg.steps + 1), unit_scale=1, smoothing=0)
    if cfg.check_time_usage:
        start_time = time.time()  # Performance tracking
    episode = 1
    for step in pbar:
        # Perform initial training (if needed)
        if step == 1:
            for _ in tqdm(range(cfg.imitation.epochs), leave=False):
                if cfg.algorithm == 'RED':
                    # Train predictor network to match random target network
                    target_estimation_update(discriminator, expert_trajectories, discriminator_optimiser,
                                             cfg.training.batch_size)
                    with torch.inference_mode():
                        discriminator.set_sigma(expert_trajectories['states'], expert_trajectories['actions'])

            if cfg.check_time_usage:
                metrics['pre_training_time'] = time.time() - start_time
                start_time = time.time()

        # Collect set of trajectories by running policy π in the environment
        with torch.inference_mode():
            policy, value = agent(state)
            action = policy.sample()
            log_prob_action = policy.log_prob(action)
            next_state, reward, terminal = env.step(action)
            extrinsic_return += reward
            trajectories.append(
                dict(states=state, actions=action, rewards=torch.tensor([reward], dtype=torch.float32),
                     terminals=torch.tensor([terminal], dtype=torch.float32), log_prob_actions=log_prob_action,
                     old_log_prob_actions=log_prob_action.detach(), values=value))
            state = next_state

        if terminal:
            # Store metrics, tensorboard and reset environment
            writer.add_scalars('reward', {'intrinsic': intrinsic_reture.mean() * 100,
                                          'extrinsic': extrinsic_return}, episode)

            pbar.set_description(f'Step: {step} | Return: {extrinsic_return}')
            state, extrinsic_return = env.reset(), 0
            episode += 1

        # Update models
        if len(trajectories) >= cfg.training.batch_size:
            # Flatten policy trajectories into a single batch
            policy_trajectories = flatten_list_dicts(trajectories)

            # Train discriminator
            if cfg.algorithm == 'GAIL':
                # Use a replay buffer of previous trajectories to prevent overfitting to current policy
                policy_trajectory_replay_buffer.append(policy_trajectories)
                policy_trajectory_replays = flatten_list_dicts(policy_trajectory_replay_buffer)
                for _ in tqdm(range(cfg.imitation.epochs), leave=False):
                    adversarial_imitation_update(cfg.algorithm, agent, discriminator, expert_trajectories,
                                                 TransitionDataset(policy_trajectory_replays),
                                                 discriminator_optimiser, cfg.training.batch_size,
                                                 cfg.imitation.r1_reg_coeff)

            # Predict rewards
            states, actions, next_states, terminals = policy_trajectories['states'], policy_trajectories[
                'actions'], torch.cat([policy_trajectories['states'][1:], next_state]), policy_trajectories[
                                                          'terminals']
            with torch.inference_mode():
                policy_trajectories['rewards'] = discriminator.predict_reward(states, actions)
            intrinsic_reture = policy_trajectories['rewards']

            # Perform PPO updates (includes GAE re-estimation with updated value function)
            for _ in tqdm(range(cfg.reinforcement.ppo_epochs), leave=False):
                ppo_update(agent, policy_trajectories, next_state, agent_optimiser, cfg.reinforcement.discount,
                           cfg.reinforcement.trace_decay, cfg.reinforcement.ppo_clip,
                           cfg.reinforcement.value_loss_coeff, cfg.reinforcement.entropy_loss_coeff,
                           cfg.reinforcement.max_grad_norm)
            trajectories, policy_trajectories = [], None

    if cfg.check_time_usage:
        metrics['training_time'] = time.time() - start_time

    # Save agent and metrics
    torch.save(agent.state_dict(), 'agent.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

    env.close()
    return sum(recent_returns) / float(cfg.evaluation.average_window)


if __name__ == '__main__':
    main()
