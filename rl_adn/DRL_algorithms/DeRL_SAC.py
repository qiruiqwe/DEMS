import itertools
import os
import threading

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Tuple, Union, List, Any
import numpy as np
from copy import deepcopy
import random
from numpy.typing import NDArray
from torch.optim import Adam
from rl_adn.DRL_algorithms.SAC import AgentSAC
# Assuming these are correctly placed or added to PYTHONPATH
from rl_adn.DRL_algorithms.utility import Config, ReplayBuffer, build_mlp
from rl_adn.DRL_algorithms.Agent import AgentBase
from rl_adn.DRL_algorithms.utility import get_optim_param
from torch.distributions.normal import Normal

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class DecoupledSquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.activation = activation
        self.act_dim = act_dim

        self.linear_list1 = nn.ModuleList([nn.Linear(obs_dim, list(hidden_sizes)[0]) for _ in range(act_dim)])
        self.linear_list2 = nn.ModuleList([nn.Linear(list(hidden_sizes)[0], list(hidden_sizes)[1]) for _ in range(act_dim)])

        self.mu_layer_list = nn.ModuleList([nn.Linear(hidden_sizes[-1], 1) for _ in range(act_dim)])
        self.log_std_layer = nn.ModuleList([nn.Linear(hidden_sizes[-1], 1) for _ in range(act_dim)])
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        for i in range(self.act_dim):
            x = self.activation()(self.linear_list1[i](obs))
            x = self.activation()(self.linear_list2[i](x))

            mu_temp = self.mu_layer_list[i](x)
            if len(mu_temp.size()) == 1:
                mu_temp = torch.unsqueeze(mu_temp, dim=1)
            log_std_temp = self.log_std_layer[i](x)
            if len(log_std_temp.size()) == 1:
                log_std_temp = torch.unsqueeze(log_std_temp, dim=1)

            if i == 0:
                mu = mu_temp
                log_std = log_std_temp
            else:
                mu = torch.cat([mu, mu_temp], dim=1)
                log_std = torch.cat([log_std, log_std_temp], dim=1)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class DecoupledBiRnnSquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, decoupled):
        super().__init__()
        self.activation = activation
        self.act_dim = act_dim
        self.rnn_type = decoupled

        self.linear_list1 = nn.ModuleList([nn.Linear(obs_dim, list(hidden_sizes)[0]) for _ in range(act_dim)])
        self.linear_list2 = nn.ModuleList([nn.Linear(list(hidden_sizes)[0], list(hidden_sizes)[1]) for _ in range(act_dim)])

        self.mu_layer_list = nn.ModuleList([nn.Linear(hidden_sizes[-1] * 2, 1) for _ in range(act_dim)])
        self.log_std_layer = nn.ModuleList([nn.Linear(hidden_sizes[-1] * 2, 1) for _ in range(act_dim)])

        if decoupled == 2:
            self.rnn_linear = nn.GRU(input_size=list(hidden_sizes)[1], hidden_size=list(hidden_sizes)[1], num_layers=2, bidirectional=True)
        else:
            self.rnn_linear = nn.LSTM(input_size=list(hidden_sizes)[1], hidden_size=list(hidden_sizes)[1], num_layers=2, bidirectional=True)
        for name, param in self.rnn_linear.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        for i in range(self.act_dim):
            x = self.activation()(self.linear_list1[i](obs))
            x = torch.unsqueeze(self.activation()(self.linear_list2[i](x)), dim=0)
            if i == 0:
                x_all = x
            else:
                x_all = torch.cat([x_all, x], dim=0)
        if len(x_all.size()) == 2:
            x_all = torch.unsqueeze(x_all, dim=1)
        if self.rnn_type == 2 or self.rnn_type == 4:
            out, _ = self.rnn_linear(x_all)
        else:
            out, (_, _) = self.rnn_linear(x_all)

        for j in range(self.act_dim):
            rnn_out_dim_j = out[j, :, :]
            mu_temp = self.mu_layer_list[j](rnn_out_dim_j)
            if len(mu_temp.size()) == 1:
                mu_temp = torch.unsqueeze(mu_temp, dim=1)
            log_std_temp = self.log_std_layer[j](rnn_out_dim_j)
            if len(log_std_temp.size()) == 1:
                log_std_temp = torch.unsqueeze(log_std_temp, dim=1)

            if j == 0:
                mu = mu_temp
                log_std = log_std_temp
            else:
                mu = torch.cat([mu, mu_temp], dim=1)
                log_std = torch.cat([log_std, log_std_temp], dim=1)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, decoupled, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space
        act_dim = action_space
        act_limit = 1 # 动作范围 默认[-1,1]

        # build policy and value functions
        if decoupled == 0:
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        elif decoupled == 1:
            self.pi = DecoupledSquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        elif decoupled >= 2:
            self.pi = DecoupledBiRnnSquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, decoupled)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()


class AgentDeRL_SAC:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, discrete_value: NDArray[np.float32],
                 gpu_id: int = 0, args: Config = Config()):
        self.num_agents = action_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.polyak = args.polyak
        self.alpha = args.per_alpha
        self.gamma = args.gamma
        self.decoupled = args.decoupled
        self.num_envs = args.num_envs
        self.batch_size = args.batch_size
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(state_dim, action_dim, self.decoupled).to(self.device)
        self.ac_targ = deepcopy(self.ac).to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=args.learning_rate)
        self.q_optimizer = Adam(self.q_params, lr=args.learning_rate)
        self.discrete_value = discrete_value
        self.explore_rate = getattr(args, 'explore_rate', 0.25)  # Initial exploration rate
        self.epsilon = self.explore_rate  # Current exploration rate, can be decayed externally

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """
        Save or load training files for the Agent.

        Args:
            cwd (str): Current Working Directory where training files are saved/loaded.
            if_save (bool): If True, save files. If False, load files.
        """
        os.makedirs(cwd, exist_ok=True)
        file_path = os.path.join(cwd, "agent.pth")

        if if_save:
            try:
                torch.save({
                    'ac_state_dict': self.ac.state_dict(),
                    'ac_targ_state_dict': self.ac_targ.state_dict(),
                    'pi_optimizer_state_dict': self.pi_optimizer.state_dict(),
                    'q_optimizer_state_dict': self.q_optimizer.state_dict()
                }, file_path)
                print(f"Saved agent to {file_path}")
            except Exception as e:
                print(f"Error saving agent to {file_path}: {e}")
        else:
            if os.path.isfile(file_path):
                try:
                    checkpoint = torch.load(file_path, map_location=self.device)
                    self.ac.load_state_dict(checkpoint['ac_state_dict'])
                    self.ac_targ.load_state_dict(checkpoint['ac_targ_state_dict'])
                    self.pi_optimizer.load_state_dict(checkpoint['pi_optimizer_state_dict'])
                    self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
                    print(f"Loaded agent from {file_path}")
                except Exception as e:
                    print(f"Error loading agent from {file_path}: {e}")
            else:
                print(f"No checkpoint found at {file_path}")


    def select_action(self, state: Tensor, if_random) -> Tensor:
        if if_random:
            return torch.rand(self.action_dim) * 2 - 1.0
        else:
            actions = self.ac.act(state, False)
            if actions.ndim != 1:
                actions = actions.flatten()
            return actions

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        states_buf = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32, device=self.device)
        actions_buf = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32, device=self.device)
        rewards_buf = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32, device=self.device)
        dones_buf = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool, device=self.device)

        ary_state = env.reset()  # shape: (num_agents, state_dim)

        for t in range(horizon_len):
            # 1. 执行动作
            tensor_state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            actions = self.select_action(state=tensor_state, if_random=if_random)

            # 2. 记录状态与动作索引
            states_buf[t] = tensor_state
            actions_buf[t] = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

            # 3. 执行动作
            next_state, reward, done, _ = env.step(actions)
            # next_state, reward, terminal, info

            # 4. 状态更新
            ary_state = env.reset() if done else next_state

            # 5. 存储奖励与 done
            rewards_buf[t] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            dones_buf[t] = torch.as_tensor(done, dtype=torch.bool, device=self.device)

        undones_buf = 1.0 - dones_buf.type(torch.float32)
        return states_buf, actions_buf, rewards_buf, undones_buf

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, buffer):
        o, a, r, d, o2 = buffer
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def compute_loss_q(self, buffer):
        o, a, r, d, o2 = buffer

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    def update_net(self, buffer: List[ReplayBuffer]) -> List[Any]:
        batch_size = self.batch_size
        update_times = int(buffer.cur_size/batch_size)

        for _ in range(update_times):
            batch = buffer.sample(batch_size)
            self.q_optimizer.zero_grad()
            loss_q, q_info = self.compute_loss_q(batch)
            loss_q.backward()
            self.q_optimizer.step()

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(batch)
            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params:
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)