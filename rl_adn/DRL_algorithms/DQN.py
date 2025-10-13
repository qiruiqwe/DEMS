import torch
from torch import nn, Tensor
from typing import Tuple, Union
import numpy as np
from copy import deepcopy
import random
from numpy.typing import NDArray

# Assuming these are correctly placed or added to PYTHONPATH
from rl_adn.DRL_algorithms.utility import Config, ReplayBuffer, build_mlp
from rl_adn.DRL_algorithms.Agent import AgentBase
from rl_adn.DRL_algorithms.utility import get_optim_param

class QNet(nn.Module):
    """
    Q-Network for DQN: Maps state to Q-values for each discrete action.

    Args:
        dims ([int]): List of integers defining the dimensions of the hidden layers.
        state_dim (int): Dimension of the state space.
        action_dim (int): Number of discrete actions.
    """
    def __init__(self, dims: [int], state_dim: int, discrete_num: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, discrete_num])
        self.discrete_num = discrete_num

    def forward(self, state: Tensor) -> Tensor:
        """
        Computes Q-values for all actions for the given state(s).

        Args:
            state (Tensor): The input state tensor.

        Returns:
            Tensor: Q-values for each discrete action. Shape: (batch_size, action_dim)
        """
        return self.net(state)

    def get_action(self, state: Tensor, epsilon: float) -> int:
        """
        Selects an action using epsilon-greedy policy.

        Args:
            state (Tensor): The input state tensor (single state). Shape: (1, state_dim)
            epsilon (float): Probability of choosing a random action.

        Returns:
            int: The index of the selected discrete action.
        """
        if torch.rand(1, device=state.device) < epsilon:
            # Explore: choose a random action index
            return torch.randint(0, self.discrete_value_num, size=(1,))
        else:
            # Exploit: choose the action with the highest Q-value
            with torch.no_grad():
                q_tensor = self.forward(state) # Shape: (1, action_dim)
                return q_tensor.argmax()


class AgentDQN(AgentBase):
    """
    Deep Q-Network (DQN) Agent.

    Attributes:
        cri (nn.Module): Online Q-network. (Uses 'cri' name for consistency with AgentBase)
        cri_target (nn.Module): Target Q-network.
        q_net_class (type): Class type for the Q-network.
        explore_rate (float): Initial epsilon for epsilon-greedy exploration.
        epsilon (float): Current epsilon value.
    """
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, discrete_value: NDArray[np.float32], gpu_id: int = 0, args: Config = Config()):
        """
        Initializes the AgentDQN.

        Args:
            net_dims ([int]): Dimensions of the hidden layers for the Q-network.
            state_dim (int): Dimension of the state space.
            action_dim (int): Number of discrete actions.
            gpu_id (int): GPU ID. Defaults to 0.
            args (Config): Configuration arguments.
        """
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.discrete_value = torch.from_numpy(discrete_value).to(torch.float32).to(self.device)
        self.act_class = getattr(self, 'act_class', QNet)
        self.cri_class = getattr(self, 'cri_class', QNet)
        # In AgentBase: act=Actor, cri=Critic. For DQN, we only need one network type (QNet).
        # We'll assign the QNet instance to `cri` and `cri_target` for consistency.
        # `act` related attributes in AgentBase will be unused.
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim,
                         discrete_value=self.discrete_value, # Pass dummy value
                         gpu_id=gpu_id, args=args)
        self.component_dim = action_dim
        # Override network initialization using QNet
        self.act = self.act_class(net_dims, state_dim, len(discrete_value)).to(self.device)
        self.act.discrete_value = self.discrete_value
        self.act.discrete_value_num = len(discrete_value)
        self.act.action_dim = self.action_dim

        self.act_target = deepcopy(self.act)

        self.act_optimizer = torch.optim.AdamW(self.act.parameters(), self.learning_rate)

        from types import MethodType  # built-in package of Python3
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)
        # self.act_optimizer = torch.optim.AdamW(self.act.parameters(), self.learning_rate)

        # Remove actor-specific parts if they exist in AgentBase's save list
        self.save_attr_names = {'act', 'act_target', 'act_optimizer'}  # Only save critic/Q-net parts

        # Epsilon-greedy exploration parameter
        self.explore_rate = getattr(args, 'explore_rate', 0.25)  # Initial exploration rate
        self.epsilon = self.explore_rate  # Current exploration rate, can be decayed externally

    def select_action(self, state: np.ndarray, current_epsilon) -> int:
        """Selects a discrete action for a single state using epsilon-greedy strategy."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_idx = self.act.get_action(state_tensor, current_epsilon)
        return action_idx

    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories for a single environment using epsilon-greedy exploration.

        Args:
            env: The Reinforcement Learning environment.
            horizon_len (int): Number of steps for exploration.
            if_random (bool, optional): If True, use purely random actions (epsilon=1.0). Defaults to False.

        Returns:
            Tuple[Tensor, ...]: A tuple containing states, actions (indices), rewards, and undones.
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        ary_state = env.reset()

        for i in range(horizon_len):
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            current_epsilon = 1.0 if if_random else self.epsilon
            action_idx = self.select_action(ary_state, current_epsilon)
            action_idxs = action_idx.repeat(self.action_dim)
            states[i] = state
            actions[i] = action_idxs

            action = self.discrete_value[action_idxs]
            ary_action = action.detach().cpu().numpy()
            next_state, reward, done, _ = env.step(ary_action)
            ary_state = env.reset() if done else next_state

            rewards[i] = reward
            dones[i] = done

        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, float]:
        """
        Update the Q-network using experiences sampled from the replay buffer.

        Args:
            buffer (ReplayBuffer): The replay buffer.

        Returns:
            Tuple[float, float]: Average critic objective (Q-loss) and zero (as there's no separate actor objective).
        """
        obj_critics = 0.0
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        assert update_times >= 1

        for _ in range(update_times):
            # get_obj_critic handles sampling, loss calculation, and PER update if enabled
            obj_critic, _ = self.get_obj_critic(buffer, self.batch_size)  # Pass buffer and batch size
            obj_critics += obj_critic.item()

            # Update the online Q-network (self.cri)
            self.optimizer_update(self.act_optimizer, obj_critic)

            # Soft update the target Q-network (self.cri_target)
            # self.soft_update(self.act_target, self.act, self.soft_update_tau)

        # Return average critic loss, actor loss is conceptually 0 for DQN
        return obj_critics / update_times, 0.0

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Compute DQN critic loss without PER."""
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)
            # Ensure rewards and undones are column vectors if needed
            if rewards.ndim == 1: rewards = rewards.unsqueeze(1)
            if undones.ndim == 1: undones = undones.unsqueeze(1)

            # DQN target: r + gamma * max_a' Q_target(s', a')
            next_q_values = self.act(next_ss)  # Q*(s', a') for all a'
            next_q_max = next_q_values.max(dim=1, keepdim=True)[0] # max_a' Q*(s', a')
            q_labels = rewards + undones * self.gamma * next_q_max

        q_values = self.act(states).gather(1, actions[:, 0].unsqueeze(1).long())
        obj_critic = self.criterion(q_values, q_labels) # Loss(Q(s,a), target)
        return obj_critic, states # Return loss and states (states might be needed by PER version)

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Compute DQN critic loss with PER."""
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, is_weights, is_indices = buffer.sample_for_per(batch_size)
            if rewards.ndim == 1: rewards = rewards.unsqueeze(1)
            if undones.ndim == 1: undones = undones.unsqueeze(1)
            actions = actions.long()
            if actions.ndim == 1: actions = actions.unsqueeze(1)

            # DQN target calculation (same as raw)
            next_q_values = self.cri_target(next_ss)
            next_q_max = next_q_values.max(dim=1, keepdim=True)[0]
            q_labels = rewards + undones * self.gamma * next_q_max

        # Current Q-values for actions taken
        q_values = self.cri(states).gather(1, actions)

        # Calculate TD errors (using reduction='none' in criterion for PER)
        # Assuming self.criterion is SmoothL1Loss(reduction='none') when PER is used (set in AgentBase)
        td_errors = self.criterion(q_values, q_labels)
        obj_critic = (td_errors * is_weights).mean() # Weighted loss

        # Update priorities in the buffer
        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, states