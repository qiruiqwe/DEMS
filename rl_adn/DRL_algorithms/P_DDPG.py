import torch
import torch.onnx
import torch.nn.functional as F
from copy import deepcopy
from torch import nn, Tensor
from typing import Tuple, Union
from rl_adn.DRL_algorithms.utility import Config, ReplayBuffer, SumTree, build_mlp, get_episode_return, get_optim_param
from rl_adn.DRL_algorithms.Agent import AgentBase
from numpy.typing import NDArray
import numpy as np

class Actor(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int, discrete_value: Tensor):
        """
        Initializes the Actor network for the DDPG algorithm.

        Args:
            dims ([int]): List of integers defining the dimensions of the hidden layers.
            state_dim (int): Dimension of the state space.
            action_dim (int): 每个组件的动作数量（这里为3）。
            num_components (int): 组件数量（这里为3）。

        Attributes:
            net: Neural network created using the specified dimensions.
            explore_noise_std: Standard deviation of exploration action noise, initialized as None.
        """
        self.discrete_value = discrete_value
        self.discrete_num = discrete_value.shape[0]  # 离散值数目
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim * self.discrete_num])
        self.explore_noise_std = None  # standard deviation of exploration action noise
        self.action_dim = action_dim

    def forward(self, state: Tensor) -> Tensor:
        """
        前向传播，输出每个组件的动作概率分布。

        Args:
            state (Tensor): 输入状态张量。

        Returns:
            Tensor: 形状为 [batch_size, NUM_COMPONENTS, ACTION_DIM] 的概率分布。
        """
        logits = self.net(state)
        # logits = logits.view(-1, self.action_dim, self.discrete_num)  # 重塑为 [batch_size, 3, 3]
        # probs = F.softmax(logits, dim=-1)  # 转换为概率分布
        return logits

    def get_action(self, state: Tensor) -> Tensor:
        """
        Computes the action for a given state with added exploration noise.

        Args:
            state (Tensor): The input state tensor.
            explore (bool): 是否进行探索。

        Returns:
            Tensor: The action tensor with added exploration noise.
        """
        logits = self.forward(state)  # [batch_size, num_components, action_dim]
        logits = logits.view(-1, self.action_dim, self.discrete_num)  # 重塑为 [batch_size, 3, 3]
        probs = F.softmax(logits, dim=-1)  # 转换为概率分布
        action_idx = torch.argmax(probs, dim=-1)
        # action_idx_cpu = action_idx.cpu()
        actions = self.discrete_value[action_idx]
        return actions

    def get_action_noise(self, state: Tensor, action_std: float) -> Tensor:
        """
        计算带噪声的动作。

        Args:
            state (Tensor): 输入状态张量，形状 [batch_size, state_dim]。
            action_std (float): 探索噪声的标准差。

        Returns:
            Tensor: 选定的离散动作，形状 [batch_size, action_dim]。
        """
        logits = self.forward(state)  # [batch_size, action_dim * discrete_num]
        logits = logits.view(-1, self.action_dim, self.discrete_num)  # 重塑为 [batch_size, 3, 3]

        # 生成高斯噪声
        noise = torch.randn_like(logits) * action_std
        noisy_logits = logits + noise  # 对 logits 直接加噪声

        # 计算带噪声的概率分布
        noisy_probs = F.softmax(noisy_logits, dim=-1)

        # 从概率分布中采样动作
        action_idx = torch.multinomial(noisy_probs.view(-1, self.discrete_num), num_samples=1).view(-1, self.action_dim)

        # 将索引映射到离散动作
        actions = self.discrete_value[action_idx]
        return actions

class Critic(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int, discrete_value: Tensor):
        """
        Initializes the Critic network for the DDPG algorithm.

        Args:
            dims ([int]): List of integers defining the dimensions of the hidden layers.
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.

        Attributes:
            net: Neural network created using the specified dimensions.
        """
        self.discrete_value = discrete_value
        self.discrete_num = discrete_value.shape[0]  # 离散值数目
        super().__init__()
        self.net = build_mlp(dims=[state_dim + action_dim, *dims, 1])

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Defines the forward pass of the Critic network.

        Args:
            value (Tensor): The input tensor combining state and action.

        Returns:
            Tensor: The output Q-value tensor.
        """

        value = torch.cat((state, action), dim=-1)  # 拼接状态和动作
        return self.net(value)  # 输出Q值


class AgentPDDPG(AgentBase):
    """
    A DDPG agent based on probability distribution for discrete actions.

    Implements the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.

    This class is responsible for the overall management of the actor and critic networks, including their initialization, updates, and interactions with the environment.

    Attributes:
        act_class: Actor class for creating the actor network.
        cri_class: Critic class for creating the critic network.
        act_target: Target actor network for stable training.
        cri_target: Target critic network for stable training.
        explore_noise_std: Standard deviation of exploration noise.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, discrete_value: NDArray[np.float32], gpu_id: int = 0, args: Config = Config()):
        """
        Initializes the AgentDDPG with the specified network dimensions, state and action dimensions, and other configurations.

        Args:
            net_dims ([int]): List of integers defining the dimensions of the hidden layers for the actor and critic networks.
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            gpu_id (int): GPU ID for running the networks. Defaults to 0.
            args (Config): Configuration object with additional settings.
        """
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        print(self.device)
        self.discrete_value = torch.from_numpy(discrete_value).to(torch.float32).to(self.device)
        self.discrete_num = discrete_value.shape[0]  # 离散值数目
        self.act_class = getattr(self, 'act_class', Actor)
        self.cri_class = getattr(self, 'cri_class', Critic)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, discrete_value=self.discrete_value, gpu_id=gpu_id, args=args)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)
        '''comapre to TD3, there is no policy noise'''
        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # standard deviation of exploration noise
        self.act.explore_noise_std = self.explore_noise_std  # assign explore_noise_std for agent.act.get_action(state)

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        """
        Updates the networks (actor and critic) using the given replay buffer.

        Args:
            buffer (ReplayBuffer): The replay buffer containing experience tuples.

        Returns:
            Tuple[float, ...]: A tuple containing the average objective values for the critic and actor updates.
        """
        obj_critics = 0.0
        obj_actors = 0.0
        # update_times = int(buffer.add_size * self.repeat_times)
        update_times = int(buffer.cur_size * self.repeat_times/self.batch_size)
        assert update_times >= 1
        for update_c in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            # 更新Actor
            # action = self.act.get_action_noise(state, self.explore_noise_std)  # policy gradient
            action = self.act.get_action(state)  # policy gradient
            obj_actor = self.cri_target(state, action).mean()  # use cri_target is more stable than cri
            obj_actors += obj_actor.item()
            self.optimizer_update(self.act_optimizer, -obj_actor)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Computes the objective for the critic network using raw experiences from the buffer.

        Args:
            buffer (ReplayBuffer): The replay buffer containing experience tuples.
            batch_size (int): The size of the batch to sample from the buffer.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the critic objective and the sampled states.
        """

        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            # Check if rewards and undones have the same dimensions as states and actions
            if rewards.dim() != states.dim():
                rewards = rewards.unsqueeze(-1)
            if undones.dim() != states.dim():
                undones = undones.unsqueeze(-1)
            '''compare with TD3 no policy noise'''
            # next_ac = self.act_target.get_action_noise(next_ss, self.explore_noise_std) # next actions
            next_ac = self.act_target.get_action(next_ss)  # next actions
            next_qs = self.cri_target(next_ss, next_ac)  # next q values
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states, actions)
        obj_critic = self.criterion(q_values,q_labels)
        return obj_critic, states

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Computes the objective for the critic network using prioritized experiences from the buffer.

        Args:
            buffer (ReplayBuffer): The replay buffer containing experience tuples.
            batch_size (int): The size of the batch to sample from the buffer.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the critic objective and the sampled states.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, is_weights, is_indices = buffer.sample_for_per(batch_size)
            # is_weights, is_indices: important sampling `weights, indices` by Prioritized Experience Replay (PER)
            # Check if rewards and undones have the same dimensions as states and actions
            if rewards.dim() != states.dim():
                rewards = rewards.unsqueeze(-1)
            if undones.dim() != states.dim():
                undones = undones.unsqueeze(-1)
            next_as = self.act_target.get_action_noise(next_ss, self.explore_noise_std)
            next_qs = self.cri_target(next_ss, next_as)  # next q values
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states, actions)
        td_errors = self.criterion(q_values, q_labels)
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, states

    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        """
        Explores the environment for a given number of steps.

        Args:
            env: The environment to be explored.
            horizon_len (int): The number of steps to explore.
            if_random (bool): Flag to determine if actions should be random. Defaults to False.

        Returns:
            [Tensor]: A list of tensors containing states, actions, rewards, and undones (not done flags).
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        ary_state = env.reset()
        for i in range(horizon_len):
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            if if_random:
                indices = torch.randint(0, self.discrete_value.size(0), (self.action_dim,), device=self.device)
                action = torch.gather(self.discrete_value, 0, indices)
            else:
                action = self.act.get_action_noise(state.unsqueeze(0), action_std=self.explore_noise_std)[0]

            states[i] = state
            actions[i] = action

            ary_action = action.detach().cpu().numpy()
            next_state, reward, done,_ = env.step(ary_action)
            ary_state = env.reset() if done else next_state

            rewards[i] = reward
            dones[i] = done

        # rewards = rewards.unsqueeze(1)
        undones = 1.0 - dones.type(torch.float32)
        # undones = (1.0 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, rewards, undones
