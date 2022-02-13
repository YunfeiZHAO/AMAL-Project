import torch
from torch import autograd
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


# Computes and stores generalised advantage estimates ψ in the set of trajectories
def compute_advantages_(trajectories, next_value, discount, trace_decay):
    reward_to_go, advantage = torch.zeros(1), torch.zeros(1)
    trajectories['rewards_to_go'], trajectories['advantages'] = torch.empty_like(
        trajectories['rewards']), torch.empty_like(trajectories['rewards'])
    for t in reversed(range(trajectories['states'].size(0))):
        reward_to_go = trajectories['rewards'][t] + (1 - trajectories['terminals'][t]) * (
                    discount * reward_to_go)  # Reward-to-go/value R
        trajectories['rewards_to_go'][t] = reward_to_go
        td_error = trajectories['rewards'][t] + (1 - trajectories['terminals'][t]) * discount * next_value - \
                   trajectories['values'][t]  # TD-error δ
        advantage = td_error + (1 - trajectories['terminals'][
            t]) * discount * trace_decay * advantage  # Generalised advantage estimate ψ
        trajectories['advantages'][t] = advantage
        next_value = trajectories['values'][t]
    # Normalise the advantages
    trajectories['advantages'] = (trajectories['advantages'] - trajectories['advantages'].mean()) / (
                trajectories['advantages'].std() + 1e-8)


# Performs one PPO update (includes GAE re-estimation)
def ppo_update(agent, trajectories, next_state, agent_optimiser, discount, trace_decay, ppo_clip, value_loss_coeff=1,
               entropy_loss_coeff=1, max_grad_norm=1):
    policy, trajectories['values'] = agent(trajectories['states'])
    trajectories['log_prob_actions'] = policy.log_prob(trajectories['actions'])
    with torch.no_grad():  # Do not differentiate through advantage calculation
        next_value = agent(next_state)[1]
        compute_advantages_(trajectories, next_value, discount,
                            trace_decay)  # Recompute rewards-to-go R and generalised advantage estimates ψ based on the current value function V

    policy_ratio = (trajectories['log_prob_actions'] - trajectories['old_log_prob_actions']).exp()
    policy_loss = -torch.min(policy_ratio * trajectories['advantages'],
                             torch.clamp(policy_ratio, min=1 - ppo_clip, max=1 + ppo_clip) * trajectories[
                                 'advantages']).mean()  # Update the policy by maximising the clipped PPO objective
    value_loss = F.mse_loss(trajectories['values'],
                            trajectories['rewards_to_go'])  # Fit value function by regression on mean squared error
    entropy_loss = -policy.entropy().mean()  # Add entropy regularisation

    agent_optimiser.zero_grad(set_to_none=True)
    (policy_loss + value_loss_coeff * value_loss + entropy_loss_coeff * entropy_loss).backward()
    clip_grad_norm_(agent.parameters(), max_grad_norm)  # Clamp norm of gradients
    agent_optimiser.step()


# Performs a target estimation update for 'RED', to have same presentation for same transitions
def target_estimation_update(discriminator, expert_trajectories, discriminator_optimiser, batch_size):
    expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True,
                                   num_workers=4)

    for expert_transition in expert_dataloader:
        expert_state, expert_action = expert_transition['states'], expert_transition['actions']

        discriminator_optimiser.zero_grad(set_to_none=True)
        prediction, target = discriminator(expert_state, expert_action)
        regression_loss = F.mse_loss(prediction, target)
        regression_loss.backward()
        discriminator_optimiser.step()


# Performs an adversarial imitation learning update
def adversarial_imitation_update(algorithm, agent, discriminator, expert_trajectories, policy_trajectories,
                                 discriminator_optimiser, batch_size, r1_reg_coeff=1):
    expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True,
                                   num_workers=4)
    policy_dataloader = DataLoader(policy_trajectories, batch_size=batch_size, shuffle=True, drop_last=True,
                                   num_workers=4)

    # Iterate over mininum of expert and policy data
    for expert_transition, policy_transition in zip(expert_dataloader, policy_dataloader):
        expert_state, expert_action, expert_next_state, expert_terminal = expert_transition['states'], \
                                                                          expert_transition['actions'], \
                                                                          expert_transition['next_states'], \
                                                                          expert_transition['terminals']
        policy_state, policy_action, policy_next_state, policy_terminal = policy_transition['states'], \
                                                                          policy_transition['actions'], \
                                                                          policy_transition['next_states'], \
                                                                          policy_transition['terminals']

        if algorithm == 'GAIL':
            D_expert = discriminator(expert_state, expert_action)
            D_policy = discriminator(policy_state, policy_action)

        # Binary logistic regression
        discriminator_optimiser.zero_grad(set_to_none=True)
        expert_loss = F.binary_cross_entropy_with_logits(D_expert, torch.ones_like(D_expert))  # Loss on "real" (expert) data
        autograd.backward(expert_loss, create_graph=True)
        r1_reg = 0
        for param in discriminator.parameters():
            r1_reg += param.grad.norm()  # R1 gradient penalty
        policy_loss = F.binary_cross_entropy_with_logits(D_policy, torch.zeros_like(D_policy))  # Loss on "fake" (policy) data
        (policy_loss + r1_reg_coeff * r1_reg).backward()
        discriminator_optimiser.step()
