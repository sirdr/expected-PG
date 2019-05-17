import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, grad
import pdb
import integration
import random

'''
    Policy is trained using any policy gradient method.
    This variant uses backward() to train the policy in a batch fashion, using the standard REINFORCE update.
'''

class Policy(nn.Module):
    def __init__(self, env, config, metrics_writer):
        super(Policy, self).__init__()
        self.clip_grad = config.clip_grad
        self.clip_actions = config.clip_actions
        self.env = env
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.action_space_high = torch.from_numpy(env.action_space.high).float()
        self.action_space_low = torch.from_numpy(env.action_space.low).float()

        self.layers = [nn.Linear(self.state_space, config.policy_layers[0])]
        for i in range(1, len(config.policy_layers)):
            self.layers.append(nn.Linear(config.policy_layers[i-1], config.policy_layers[i]))
        self.layers.append(nn.Linear(config.policy_layers[-1], self.action_space))
        self.layers = nn.ModuleList(self.layers)

        if config.learn_std:
            self.log_std = torch.nn.Parameter(torch.tensor([np.log(config.action_std)], dtype=torch.float32), requires_grad=True)
        else:
            self.log_std = torch.tensor([np.log(config.action_std)], dtype=torch.float32)

        self.gamma = config.gamma
        self.normalize_advantages = config.normalize_advantages

        self.optimizer = optim.Adam(self.parameters(), lr=config.policy_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.policy_lr_step_size, gamma=config.policy_lr_decay)

        self.metrics_writer = metrics_writer

    def forward(self, out):
        for layer in self.layers[:-1]:
            out = layer(out)
            out = F.relu(out)
        out = self.layers[-1](out)
        if self.clip_actions:
            out = torch.max(torch.min(out, self.action_space_high), self.action_space_low)
        else:
            out = (F.tanh(out) + 1) * .5 * (self.action_space_high - self.action_space_low) + self.action_space_low
        return out

    def get_action(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_mean = self.forward(state)
        std = torch.exp(self.log_std)
        dist = torch.distributions.normal.Normal(action_mean, std)
        sample = dist.sample().numpy()
        return np.clip(sample, self.action_space_low.numpy(), self.action_space_high.numpy())

    def get_actions(self, state, n):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_means = self.forward(state).repeat(n, 1)
        std = torch.exp(self.log_std)
        dist = torch.distributions.normal.Normal(action_means, std)
        sample = dist.sample()
        return torch.max(torch.min(sample, self.action_space_high), self.action_space_low)


class PolicyReinforce(Policy):
    def __init__(self, env, config, metrics_writer):
        super(PolicyReinforce, self).__init__(env, config, metrics_writer)
        self.normalize_advantages = config.normalize_advantages

    def compute_advantages(self, states, rewards, vcritic=None):
        g = np.array(rewards)
        n_transitions = len(g)
        g = (self.gamma ** np.arange(n_transitions)) * g
        advantages = np.cumsum(g[::-1])[::-1] / (self.gamma ** np.arange(n_transitions))
        if vcritic is not None:
            advantages -= vcritic(torch.from_numpy(np.vstack(states)).float()).detach().numpy().flatten()
        return advantages

    def compute_advantages_batch(self, states, rewards, vcritic=None):
        advantages = []
        for k in range(len(states)):
            ep_states, ep_rewards = states[k][:-1], rewards[k]
            g = np.array(ep_rewards)
            n_transitions = len(g)
            g = (self.gamma ** np.arange(n_transitions)) * g
            ep_advantages = np.cumsum(g[::-1])[::-1] / (self.gamma ** np.arange(n_transitions))
            if vcritic is not None:
                ep_advantages -= vcritic(torch.from_numpy(np.vstack(ep_states)).float()).detach().numpy().flatten()
            advantages.append(ep_advantages)
        return np.concatenate(advantages)

    def apply_gradient_episode(self, ep_states, ep_actions, ep_rewards, episode, vcritic):

        self.optimizer.zero_grad()
        n_states = len(ep_states)
        states = torch.tensor(np.array([state for state in ep_states[:-1]])).float()
        actions = torch.tensor(np.array([action for action in ep_actions])).float()
        advantages = torch.tensor(self.compute_advantages(states, ep_rewards, vcritic=vcritic)).float()

        self.metrics_writer.write_metric(episode, "average_advantage", torch.mean(advantages))
        self.metrics_writer.write_metric(episode, "std_advantage", torch.std(advantages))

        if(self.normalize_advantages):
            advantages = (advantages - torch.mean(advantages)) / (0.1 + torch.std(advantages))

        std = torch.exp(self.log_std)

        action_means = self.forward(states)
        log_probs = torch.sum(torch.distributions.normal.Normal(action_means, std).log_prob(actions), dim=1)

        loss = - torch.dot(log_probs, advantages.detach()) / n_states
        loss.backward()

        for name, param in self.named_parameters():
            self.metrics_writer.write_metric(episode, f"grad_norm_{name}", torch.norm(param.grad))

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm(self.parameters(), self.clip_grad)

        self.optimizer.step()
        self.scheduler.step()

    def apply_gradient_batch(self, states, actions, rewards, batch, vcritic):

        self.optimizer.zero_grad()
        n_states = sum([len(ep_states[:-1]) for ep_states in states])
        advantages = torch.tensor(self.compute_advantages_batch(states, rewards, vcritic=vcritic)).float()
        states = torch.tensor(np.array([state for ep_states in states for state in ep_states[:-1]])).float()
        actions = torch.tensor(np.array([action for ep_actions in actions for action in ep_actions])).float()

        self.metrics_writer.write_metric(batch, "average_advantage", torch.mean(advantages))
        self.metrics_writer.write_metric(batch, "std_advantage", torch.std(advantages))

        if(self.normalize_advantages):
            advantages = (advantages - torch.mean(advantages)) / (0.1 + torch.std(advantages))

        std = torch.exp(self.log_std)
        log_probs = torch.sum(torch.distributions.normal.Normal(self.forward(states), std).log_prob(actions), dim=1)

        loss = - torch.dot(log_probs, advantages.detach()) / n_states
        loss.backward()

        for name, param in self.named_parameters():
            self.metrics_writer.write_metric(batch, f"grad_norm_{name}", torch.norm(param.grad))

        self.optimizer.step()

'''
    Policy is trained using any policy gradient method.
    This variant uses backward() to train the policy in an episodic fashion, using the standard REINFORCE update.
'''
class PolicyMC(Policy):
    def __init__(self, env, config, metrics_writer, num_actions=100):
        super(PolicyMC, self).__init__(env, config, metrics_writer)
        self.n_samples_per_state = num_actions

    def apply_gradient_episode(self, ep_states, ep_actions, ep_rewards, episode, qcritic, vcritic):

        self.optimizer.zero_grad()

        n_states = len(ep_states)
        states = torch.tensor(np.array([state for state in ep_states[:-1]])).float()
        action_means = self.forward(states).repeat(self.n_samples_per_state, 1)
        action_means = torch.max(torch.min(action_means, self.action_space_high), self.action_space_low)

        std = torch.exp(self.log_std)
        dist = torch.distributions.normal.Normal(action_means, std)
        sample = dist.sample()
        actions = torch.max(torch.min(sample, self.action_space_high), self.action_space_low)

        advantages = qcritic(states.repeat(self.n_samples_per_state, 1), actions).flatten().detach()
        self.metrics_writer.write_metric(episode, "average_q", torch.mean(advantages))
        if vcritic is not None:
            v_values = vcritic(states.repeat(self.n_samples_per_state,1)).flatten().detach()
            self.metrics_writer.write_metric(episode, "average_v", torch.mean(v_values))
            advantages -= v_values

        self.metrics_writer.write_metric(episode, "average_advantage", torch.mean(advantages))
        self.metrics_writer.write_metric(episode, "std_advantage", torch.std(advantages))

        if(self.normalize_advantages):
            advantages = (advantages - torch.mean(advantages)) / (0.1 + torch.std(advantages))

        self.metrics_writer.write_metric(episode, "advantage_range", torch.max(advantages) - torch.min(advantages))

        log_probs = torch.sum(torch.distributions.normal.Normal(action_means, std).log_prob(actions), dim=1)

        self.metrics_writer.write_metric(episode, "avg_log_prob", torch.mean(log_probs))
        self.metrics_writer.write_metric(episode, "max_log_prob", torch.max(log_probs))
        self.metrics_writer.write_metric(episode, "min_log_prob", torch.min(log_probs))
        self.metrics_writer.write_metric(episode, "avg_action_mean", torch.mean(action_means))
        self.metrics_writer.write_metric(episode, "min_action_mean", torch.min(action_means))
        self.metrics_writer.write_metric(episode, "max_action_mean", torch.max(action_means))

        loss = - torch.dot(log_probs, advantages.detach()) / (self.n_samples_per_state * n_states)
        loss.backward()

        for name, param in self.named_parameters():
            self.metrics_writer.write_metric(episode, f"grad_norm_{name}", torch.norm(param.grad))

        # torch.nn.utils.clip_grad_norm(self.parameters(), 2)
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm(self.parameters(), self.clip_grad)

        self.optimizer.step()
        self.scheduler.step()

    def apply_gradient_batch(self, states, actions, rewards, batch, qcritic, vcritic):

        self.optimizer.zero_grad()

        n_states = sum([len(ep_states[:-1]) for ep_states in states])
        states = torch.tensor(np.array([state for ep_states in states for state in ep_states[:-1]])).float()
        action_means = self.forward(states).repeat(self.n_samples_per_state, 1)

        std = torch.exp(self.log_std)
        dist = torch.distributions.normal.Normal(action_means, std)
        sample = dist.sample()
        actions = torch.max(torch.min(sample, self.action_space_high), self.action_space_low)

        advantages = qcritic(states.repeat(self.n_samples_per_state, 1), actions).flatten().detach()
        if vcritic is not None:
            advantages -= vcritic(states.repeat(self.n_samples_per_state,1)).flatten().detach()

        self.metrics_writer.write_metric(batch, "average_advantage", torch.mean(advantages))
        self.metrics_writer.write_metric(batch, "std_advantage", torch.std(advantages))

        if(self.normalize_advantages):
            advantages = (advantages - torch.mean(advantages)) / (0.1 + torch.std(advantages))

        log_probs = torch.sum(torch.distributions.normal.Normal(action_means, std).log_prob(actions), dim=1)

        loss = - torch.dot(log_probs, advantages.detach()) / self.n_samples_per_state / n_states
        loss.backward()

        for name, param in self.named_parameters():
            self.metrics_writer.write_metric(batch, f"grad_norm_{name}", torch.norm(param.grad))

        self.optimizer.step()

'''
    Policy is trained using expected policy gradients through numerical integration.
'''
class PolicyIntegration(Policy):
    def __init__(self, env, config, metrics_writer):
        super(PolicyIntegration, self).__init__(env, config, metrics_writer)

    def apply_gradient_episode(self, ep_states, ep_actions, ep_rewards, episode, qcritic, vcritic=None):

        self.optimizer.zero_grad()
        grads = {}
        for name, param in self.named_parameters():
            grads[name] = torch.zeros_like(param)

        states = [state for state in ep_states[:-1]]
        n_actions = len(ep_actions)

        std = torch.exp(self.log_std)

        # Manually compute gradients
        for i in range(n_actions):

            state = states[i]
            action_means = self.forward(torch.from_numpy(state).float())

            for name, param in self.named_parameters():
                tensor_state = torch.from_numpy(state).float()
                dist = torch.distributions.normal.Normal(self.forward(tensor_state), std)
                fun = lambda a : (grad(torch.exp(dist.log_prob(torch.from_numpy(a).float())), param, retain_graph=True)[0] * (qcritic(tensor_state, torch.from_numpy(a).float()) - vcritic(tensor_state))).detach().numpy()
                estimate = integration.compute_integral_asr(fun, self.action_space_low.numpy(), self.action_space_high.numpy(), 0.01)
                grads[name] -= torch.from_numpy(estimate).float()

        # Manually assign gradients to the right tensors.
        for name, param in self.named_parameters():
            param.grad = grads[name]
            self.metrics_writer.write_metric(episode, f"grad_norm_{name}", torch.norm(param.grad))
            print(name, param.grad)

        # Take an optimizer step.
        self.optimizer.step()
        self.scheduler.step()

class PolicyIntegrationTrapezoidal(Policy):
    def __init__(self, env, config, metrics_writer, num_actions=1000):
        super(PolicyIntegrationTrapezoidal, self).__init__(env, config, metrics_writer)
        self.n_samples_per_state = config.n_samples_per_state
        self.num_actions = num_actions

        self.actions = torch.tensor(np.stack(np.meshgrid(*[np.linspace(self.action_space_low[k], self.action_space_high[k], num_actions) for k in range(self.action_space_low.shape[0])], indexing='ij'), axis=-1).reshape(-1, self.action_space_low.shape[0])).float()
        self.total_actions = self.actions.shape[0]

        self.weight = ((self.action_space_high - self.action_space_low) / (num_actions - 1)).prod()

    def apply_gradient_episode(self, ep_states, ep_actions, ep_rewards, episode, qcritic, vcritic=None):

        self.optimizer.zero_grad()

        states = np.array(ep_states[:-1])
        num_states = states.shape[0]

        states = np.reshape(np.tile(states, len(self.actions)), (len(self.actions)*num_states, -1))

        states = torch.tensor(states).float()

        actions = self.actions.repeat(num_states,1)

        action_means = self.forward(states)

        advantages = qcritic(states, actions).flatten().detach() - vcritic(states).flatten().detach()

        self.metrics_writer.write_metric(episode, "average_advantage", torch.mean(advantages))
        self.metrics_writer.write_metric(episode, "std_advantage", torch.std(advantages))
        if(self.normalize_advantages):
            advantages = (advantages - torch.mean(advantages)) / (.1 + torch.std(advantages))

        std = torch.exp(self.log_std)
        log_probs = torch.sum(torch.distributions.normal.Normal(action_means, std).log_prob(actions), dim=1).flatten()
        probs = torch.exp(log_probs)

        integrand = probs * advantages
        integrand_reshaped = torch.reshape(integrand, [-1, self.total_actions])
        integrand_reshaped_avg = (integrand_reshaped[:, :-1] + integrand_reshaped[:, 1:])/2.0
        integrand_avg = torch.reshape(integrand_reshaped_avg, [-1, 1])
        weighted_integrand = self.weight*integrand_avg

        loss = -torch.sum(weighted_integrand) / (num_states * self.total_actions)
        loss.backward()

        for name, param in self.named_parameters():
            self.metrics_writer.write_metric(episode, f"grad_norm_{name}", torch.norm(param.grad))

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm(self.parameters(), self.clip_grad)

        self.optimizer.step()
        self.scheduler.step()
