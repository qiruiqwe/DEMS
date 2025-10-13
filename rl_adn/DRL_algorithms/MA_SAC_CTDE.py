import torch
import numpy as np
import torch.onnx
import torch.nn as nn
import copy as cp
from copy import deepcopy
import os
from torch import nn, Tensor
from typing import Tuple, Union
from rl_adn.DRL_algorithms.utility import Config, ReplayBuffer, SumTree, build_mlp, get_episode_return, get_optim_param
from rl_adn.DRL_algorithms.Agent import AgentBase
from numpy.typing import NDArray
import numpy as np

'''
    Centralized Training with Decentralized Execution, CTDE
'''
class ActorSAC(nn.Module):
    """
    Actor network for Soft Actor-Critic (SAC) algorithm.

    Attributes:
        enc_s (nn.Module): Encoder network for state input.
        dec_a_avg (nn.Module): Decoder network for action mean.
        dec_a_std (nn.Module): Decoder network for action log standard deviation.
        log_sqrt_2pi (float): Logarithm of the square root of 2π, a constant used in calculations.
        soft_plus (nn.Softplus): Softplus activation function.

    Methods:
        forward(state): Computes the action for a given state.
        get_action(state): Computes the action for exploration.
        get_action_logprob(state): Computes the action and its log probability for a given state.
    """
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        """
        Initializes the Actor network for SAC.

        Args:
            dims ([int]): List of integers defining the dimensions of the hidden layers.
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
        """

        super().__init__()
        self.enc_s = build_mlp(dims=[state_dim, *dims])  # encoder of state
        self.dec_a_avg = build_mlp(dims=[dims[-1], action_dim])  # decoder of action mean
        self.dec_a_std = build_mlp(dims=[dims[-1], action_dim])  # decoder of action log_std
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.soft_plus = nn.Softplus()
        setattr(self, 'act', self.dec_a_avg)
        self.save_attr_names = {'act'}

    def forward(self, state: Tensor) -> Tensor:
        """
        Performs a forward pass through the network to compute the action for a given state.

        Args:
            state (Tensor): The input state tensor.

        Returns:
            Tensor: The output tensor representing the action.
        """
        state_tmp = self.enc_s(state)  # temporary tensor of state
        return self.dec_a_avg(state_tmp).tanh()  # action

    def get_action(self, state: Tensor) -> Tensor:
        """
        Computes the action for exploration by adding noise to the action mean.
        This method is typically used during training to encourage exploration.

        Args:
            state (Tensor): The input state tensor for which the action needs to be computed.

        Returns:
            Tensor: The computed action with added exploration noise.
        """
        state_tmp = self.enc_s(state)  # temporary tensor of state
        action_avg = self.dec_a_avg(state_tmp)
        action_std = self.dec_a_std(state_tmp).clamp(-20, 2).exp()

        noise = torch.randn_like(action_avg, requires_grad=False)
        action = (action_avg + action_std * noise).tanh()
        return action  # action (re-parameterize)

    def get_action_logprob(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the action and its log probability for a given state. This method is
        used for SAC's policy update, where both the action and its log probability are required.

        Args:
            state (Tensor): The input state tensor for which the action and log probability are computed.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the computed action and its log probability.
        """
        state_tmp = self.enc_s(state)  # temporary tensor of state
        action_log_std = self.dec_a_std(state_tmp).clamp(-20, 2)
        action_std = action_log_std.exp()
        action_avg = self.dec_a_avg(state_tmp)

        '''add noise to a_noise in stochastic policy'''
        noise = torch.randn_like(action_avg, requires_grad=True)
        a_noise = action_avg + action_std * noise

        '''Log prob of Gaussian distribution'''
        log_prob = -action_log_std - noise.pow(2) * 0.5 - self.log_sqrt_2pi

        '''fix logprob by adding the derivative of y=tanh(x)'''
        log_prob -= torch.log(1.0 - a_noise.tanh().pow(2) + 1e-6).sum(dim=1, keepdim=True)
        return a_noise.tanh(), log_prob.sum(1, keepdim=True)

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        cwd: Current Working Directory. ElegantRL save training files in CWD.
        if_save: True: save files. False: load files.
        """
        assert self.save_attr_names.issuperset({'act'})

        for attr_name in self.save_attr_names:
            file_path = f"{cwd}/{attr_name}.pth"
            print(file_path)
            if if_save:
                torch.save(getattr(self, attr_name), file_path)
            elif os.path.isfile(file_path):
                setattr(self, attr_name, torch.load(file_path, map_location=self.device))
            else:
                raise FileNotFoundError(f"Loading failed: {file_path} does not exist!")

# --- Define the Centralized Critic ---
class CentralizedCriticTwin(nn.Module):
    """
    Centralized Twin Critic network for MA-SAC.
    Takes joint states and joint actions as input.
    """
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        """
        Initializes the Centralized Twin Critic network.

        Args:
            dims ([int]): List of integers defining the dimensions of the hidden layers.
            state_dim (int): Sum of state dimensions of all agents.
            action_dim (int): Sum of action dimensions of all agents.
        """
        super().__init__()
        # Input dimension is the concatenation of all states and all actions
        input_dim = state_dim + action_dim
        self.enc_sa = build_mlp(dims=[input_dim, *dims])  # encoder of joint state and action
        self.dec_q1 = build_mlp(dims=[dims[-1], 1])  # decoder of Q value 1
        self.dec_q2 = build_mlp(dims=[dims[-1], 1])  # decoder of Q value 2

    def forward(self, joint_state: Tensor, joint_action: Tensor) -> Tensor:
        """
        Computes the first Q value for a given joint state-action pair.

        Args:
            joint_state (Tensor): Concatenated states of all agents.
            joint_action (Tensor): Concatenated actions of all agents.

        Returns:
            Tensor: The output tensor representing the first Q value.
        """
        value = torch.cat((joint_state, joint_action), dim=1)
        sa_tmp = self.enc_sa(value)
        return self.dec_q1(sa_tmp)  # Q value

    def get_q1_q2(self, joint_state: Tensor, joint_action: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes both Q values for a given joint state-action pair.

        Args:
            joint_state (Tensor): Concatenated states of all agents.
            joint_action (Tensor): Concatenated actions of all agents.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the two Q values.
        """
        value = torch.cat((joint_state, joint_action), dim=1)
        sa_tmp = self.enc_sa(value)
        return self.dec_q1(sa_tmp), self.dec_q2(sa_tmp)  # two Q values


class AgentMASAC_CTDE(AgentBase):
    """
    Centralized Training with Decentralized Execution, CTDE
    Soft Actor-Critic (SAC) agent implementation.

    Attributes:
        act_class (type): Class type for the actor network.
        cri_class (type): Class type for the critic network.
        cri_target (nn.Module): Target critic network for stable training.
        alpha_log (Tensor): Logarithm of the temperature parameter alpha.
        alpha_optimizer (torch.optim.Optimizer): Optimizer for alpha.
        target_entropy (float): Target entropy for policy optimization.

    Methods:
        explore_one_env(env, horizon_len, if_random): Explores an environment for a given horizon length.
        update_net(buffer): Updates the networks using the given replay buffer.
        get_obj_critic_raw(buffer, batch_size): Computes the raw objective for the critic.
        get_obj_critic_per(buffer, batch_size): Computes the PER-adjusted objective for the critic.
    """
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, discrete_value: NDArray[np.float32], gpu_id: int = 0, args: Config = Config()):

        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.discrete_value = torch.from_numpy(discrete_value).to(torch.float32).to(self.device)
        self.agent_nums = action_dim
        per_agent_action_dim = action_dim // self.agent_nums
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.act_class = getattr(self, 'act_class', ActorSAC)
        self.cri_class = getattr(self, 'cri_class', CentralizedCriticTwin)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim,
                         discrete_value=self.discrete_value,  # Pass dummy value
                         gpu_id=gpu_id, args=args)

        # Create Decentralized Actors (one per agent)
        self.acts = nn.ModuleList()
        self.act_optimizers = []
        for i in range(self.agent_nums):
            act = ActorSAC(net_dims, state_dim, per_agent_action_dim).to(self.device)
            self.acts.append(act)
            self.act_optimizers.append(torch.optim.Adam(act.parameters(), lr=self.learning_rate))

        # Create Centralized Critic
        self.cri = CentralizedCriticTwin(net_dims, self.state_dim, self.action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        # Temperature parameter Alpha (shared)
        self.alpha_log = torch.tensor(-1.0 * self.agent_nums, dtype=torch.float32, requires_grad=True,
                                      device=self.device)
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), lr=self.learning_rate)
        self.target_entropy = -self.action_dim  # 在每个代理动作维度为 1，采用单代理 SAC 的标准目标熵公式−adim_i,并对所有代理求和.


        # self.alpha_log = torch.tensor((-1,), dtype=torch.float32, requires_grad=True, device=self.device)  # trainable
        # self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), lr=self.learning_rate)
        # # here target entropy can be negaticve
        # self.target_entropy = np.log(action_dim)

    def select_actions(self, state: Tensor, if_random: bool = True) -> Tensor:
        """
        Selects actions for all agents based on their local observations.

        Args:
           state (Tensor): Shared state tensor for all agents.
                          Shape: (num_envs, state_dim) or (state_dim,)
        Returns:
            Tensor: Concatenated action tensor for all agents.
        """
        state = state.to(self.device) # Ensure state is on the correct device
        if if_random:
            actions = torch.rand(self.action_dim, device=self.device) * 2 - 1.0
        else:
            actions = torch.zeros(self.agent_nums, device=self.device)
            for i, act in enumerate(self.acts):
                action_value = act.get_action(state)
                actions[i] = action_value.detach()  # Only take scalar
        return actions

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        """
        Updates the networks (actor, critic, and temperature parameter) using experiences from the replay buffer.

        This method performs the core updates for the SAC algorithm, including updating the critic network, the temperature parameter for entropy maximization, and the actor network.

        Args:
            buffer (ReplayBuffer): The replay buffer containing experiences for training.

        Returns:
            Tuple[float, float, float]: A tuple containing the average objective values for the critic, actor, and alpha (temperature parameter) updates.
        """
        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        alphas = 0.0

        update_times = int(buffer.cur_size * self.repeat_times/self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            obj_critics += obj_critic.item()

            '''objective of actor'''
            state = state.detach()  # 显式确保 state 不需要梯度
            alpha = self.alpha_log.exp().detach()

            for i in range(self.agent_nums):
                # 生成联合动作，第 i 个代理需要梯度，其他不需要
                next_actions_tensor = torch.zeros(self.batch_size, self.agent_nums).to(self.device)
                next_logprobs_tensor = torch.zeros(self.batch_size, self.agent_nums).to(self.device)
                for j in range(self.agent_nums):
                    if j == i:
                        next_action_j, next_logprob_j = self.acts[j].get_action_logprob(state)
                    else:
                        with torch.no_grad():
                            next_action_j, next_logprob_j = self.acts[j].get_action_logprob(state)
                    next_actions_tensor[:, j] = next_action_j.squeeze(-1)
                    next_logprobs_tensor[:, j] = next_logprob_j.squeeze(-1)

                current_logprobs_sum = next_logprobs_tensor.sum(dim=1, keepdim=True)
                q_actor = self.cri(state, next_actions_tensor)
                obj_actor_i = (alpha * current_logprobs_sum - q_actor).mean()
                obj_actors += obj_actor_i.item()

                self.act_optimizers[i].zero_grad()
                (-obj_actor_i).backward()
                nn.utils.clip_grad_norm_(self.acts[i].parameters(), max_norm=1.0)
                self.act_optimizers[i].step()

            '''objective of alpha'''
            obj_alpha = (self.alpha_log * (-current_logprobs_sum + self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optimizer, obj_alpha)
            alphas += alpha.item()

        return obj_critics / update_times, obj_actors / update_times, alphas / update_times

    def update_net_before(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        """
        联合优化
        Updates the networks (actor, critic, and temperature parameter) using experiences from the replay buffer.

        This method performs the core updates for the SAC algorithm, including updating the critic network, the temperature parameter for entropy maximization, and the actor network.

        Args:
            buffer (ReplayBuffer): The replay buffer containing experiences for training.

        Returns:
            Tuple[float, float, float]: A tuple containing the average objective values for the critic, actor, and alpha (temperature parameter) updates.
        """
        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        alphas = 0.0

        update_times = int(buffer.cur_size * self.repeat_times/self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            obj_critics += obj_critic.item()

            '''objective of actor'''
            state = state.detach()  # 显式确保 state 不需要梯度
            next_actions_tensor = torch.zeros(self.batch_size, self.agent_nums).to(self.device)
            next_logprobs_tensor = torch.zeros(self.batch_size, self.agent_nums).to(self.device)
            for i in range(self.agent_nums):
                next_action_i, next_logprob_i = self.acts[i].get_action_logprob(state)
                next_actions_tensor[:, i] = next_action_i.squeeze(-1)
                next_logprobs_tensor[:, i] = next_logprob_i.squeeze(-1)
            current_logprobs_sum = next_logprobs_tensor.sum(dim=1, keepdim=True)

            q_actor = self.cri(state, next_actions_tensor)
            alpha = self.alpha_log.exp().detach()
            obj_actor = (alpha * current_logprobs_sum - q_actor).mean()
            obj_actors += obj_actor.item()
            for opt in self.act_optimizers:
                opt.zero_grad()  # 清零每个优化器的梯度
            (-obj_actor).backward()  # 单次反向传播，计算所有演员的梯度
            for opt in self.act_optimizers:
                opt.step()  # 更新每个演员的参数

            '''objective of alpha'''
            obj_alpha = (self.alpha_log * (-current_logprobs_sum + self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optimizer, obj_alpha)
            alphas += alpha.item()

        return obj_critics / update_times, obj_actors / update_times, alphas / update_times

    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        """
        Explores a given environment for a specified horizon length.

        This method is used for collecting experiences by interacting with the environment. It can operate in either a random action mode or a policy-based action mode.

        Args:
            env: The environment to be explored.
            horizon_len (int): The number of steps to explore the environment.
            if_random (bool): If True, actions are chosen randomly. If False, actions are chosen based on the current policy.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: A tuple containing states, actions, rewards, and undones (indicating whether the episode has ended) collected during the exploration.
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        ary_state = env.reset()

        for i in range(horizon_len):
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            action = self.select_actions(state.unsqueeze(0), if_random)

            states[i] = state
            actions[i] = action

            ary_action = action.cpu().numpy()
            next_state, reward, done,_ = env.step(ary_action)
            ary_state = env.reset() if done else next_state

            rewards[i] = reward
            dones[i] = done

        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def get_obj_critic_raw(self, buffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Computes the raw objective for the critic using experiences from the buffer.

        This method calculates the loss for the critic network based on the sampled experiences. It does not use Prioritized Experience Replay (PER) adjustments.

        Args:
            buffer (ReplayBuffer): The replay buffer containing experiences for training.
            batch_size (int): The size of the batch to sample from the buffer.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the critic loss and the sampled states.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            if rewards.dim() != states.dim():
                rewards = rewards.unsqueeze(-1)
            if undones.dim() != states.dim():
                undones = undones.unsqueeze(-1)

            '''Get next actions and logprobs for *all* agents from their policies'''
            next_actions_tensor = torch.zeros(batch_size, self.action_dim).to(self.device)
            next_logprobs_tensor = torch.zeros(batch_size, self.action_dim).to(self.device)
            for i in range(self.agent_nums):
                next_action_i, next_logprob_i = self.acts[i].get_action_logprob(next_ss)
                next_actions_tensor[:, i] = next_action_i.squeeze(-1)
                next_logprobs_tensor[:, i] = next_logprob_i.squeeze(-1)

            '''here is how to calculate the next qs and q labels'''
            next_q_target = torch.min(*self.cri_target.get_q1_q2(states.to(self.device), next_actions_tensor))
            next_logprobs_sum = next_logprobs_tensor.sum(dim=1, keepdim=True)
            alpha = self.alpha_log.exp()
            q_target = rewards + undones * self.gamma * (next_q_target - next_logprobs_sum * alpha)

        '''Calculate current Q values using the centralized critic'''
        q1, q2 = self.cri.get_q1_q2(states, actions)
        '''Calculate critic loss'''
        obj_critic = self.criterion(q1, q_target) + self.criterion(q2, q_target)  # twin critics

        return obj_critic.mean(), states

