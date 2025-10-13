import threading

import torch
from torch import nn, Tensor
from typing import Tuple, Union, List, Any
import numpy as np
from copy import deepcopy
import random
from numpy.typing import NDArray

from rl_adn.DRL_algorithms.DDPG import AgentDDPG
# Assuming these are correctly placed or added to PYTHONPATH
from rl_adn.DRL_algorithms.utility import Config, ReplayBuffer, build_mlp
from rl_adn.DRL_algorithms.Agent import AgentBase
from rl_adn.DRL_algorithms.utility import get_optim_param


class AgentMADDPG:

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, discrete_value: NDArray[np.float32],
                 gpu_id: int = 0, args: Config = Config()):
        """
        Initializes the AgentDQN.

        Args:
            net_dims ([int]): Dimensions of the hidden layers for the Q-network.
            state_dim (int): Dimension of the state space.
            action_dim (int): Number of discrete actions.
            gpu_id (int): GPU ID. Defaults to 0.
            args (Config): Configuration arguments.
        """
        self.num_agents = action_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.agents = [
            AgentDDPG(net_dims, state_dim, 1, discrete_value, gpu_id, args) for _ in range(self.num_agents)
        ]
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.discrete_value = discrete_value
        self.explore_rate = getattr(args, 'explore_rate', 0.25)  # Initial exploration rate
        self.epsilon = self.explore_rate  # Current exploration rate, can be decayed externally

    def select_action(self, state: Tensor, if_random) -> int:
        if if_random:
            actions = torch.rand(self.action_dim) * 2 - 1.0
        else:
            actions = torch.zeros(self.num_agents)
            for i, agent in enumerate(self.agents):
                action_value = agent.act.get_action(state)
                actions[i] = action_value  # Only take scalar
        return actions

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        state_dim = self.state_dim
        num_agents = self.num_agents

        states_buf = torch.zeros((horizon_len, num_agents, state_dim), dtype=torch.float32, device=self.device)
        actions_buf = torch.zeros((horizon_len, num_agents, 1), dtype=torch.float32, device=self.device)
        rewards_buf = torch.zeros((horizon_len, num_agents), dtype=torch.float32, device=self.device)
        dones_buf = torch.zeros((horizon_len, num_agents), dtype=torch.bool, device=self.device)

        ary_state = env.reset()  # shape: (num_agents, state_dim)

        for t in range(horizon_len):
            # 1. 动作索引选择（不含离散动作转换）
            tensor_state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            actions_1_5 = self.select_action(state=tensor_state, if_random=if_random)
            actions = actions_1_5.reshape(self.num_agents, 1)

            # 2. 记录状态与动作索引（原始整数动作）
            states_buf[t] = tensor_state
            actions_buf[t] = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

            # 3. 执行动作
            next_state, reward, done, _ = env.step(actions_1_5)

            # 4. 状态更新
            ary_state = env.reset() if done else next_state

            # 5. 存储奖励与 done
            rewards_buf[t] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            dones_buf[t] = torch.as_tensor(done, dtype=torch.bool, device=self.device)

        undones_buf = 1.0 - dones_buf.type(torch.float32)
        return states_buf, actions_buf, rewards_buf, undones_buf

    def update_net_single(self, buffer: List[ReplayBuffer]) -> List[Any]:
        for agent, buffer in zip(self.agents, buffer):
            agent.update_net(buffer)

    def update_net(self, buffer: List[ReplayBuffer]) -> List[Any]:
        results = []
        threads = []

        # 定义每个代理的更新函数
        def update_agent(agent, buffer, result_list):
            result = agent.update_net(buffer)
            result_list.append(result)

        # 为每个代理创建线程
        for agent, buf in zip(self.agents, buffer):
            result_list = []
            thread = threading.Thread(target=update_agent, args=(agent, buf, result_list))
            threads.append(thread)
            results.append(result_list)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 收集结果
        return [result[0] for result in results if result]