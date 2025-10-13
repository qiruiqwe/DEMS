import time
import logging
import os
import sys

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
print(parent_dir)
sys.path.append(parent_dir)
import argparse
import numpy as np
from rl_adn.environments.env import PowerNetEnv, env_config
import torch
from rl_adn.DRL_algorithms.Agent import AgentDDPG, AgentSAC, AgentBase, AgentTD3
from rl_adn.DRL_algorithms.DDPG_single_action import AgentDDPG_single
from rl_adn.DRL_algorithms.PPO import AgentPPO
from rl_adn.DRL_algorithms.SAC import AgentSAC
from rl_adn.DRL_algorithms.TD3 import AgentTD3
from rl_adn.DRL_algorithms.DQN import AgentDQN
from rl_adn.DRL_algorithms.P_DDPG import AgentPDDPG
from rl_adn.DRL_algorithms.P_SAC import AgentPSAC
from rl_adn.DRL_algorithms.P_TD3 import AgentPTD3
from rl_adn.DRL_algorithms.DDPG import AgentDDPG
from rl_adn.DRL_algorithms.utility import Config, ReplayBuffer, SumTree, build_mlp, get_episode_return, get_optim_param, \
    get_test_return
from rl_adn.DRL_algorithms.MA_DDPG import AgentMADDPG
from rl_adn.DRL_algorithms.MA_SAC import AgentMASAC
from rl_adn.DRL_algorithms.MA_TD3 import AgentMATD3
from rl_adn.DRL_algorithms.MA_DQN import AgentMADQN
from rl_adn.DRL_algorithms.MA_DDQN import AgentMADDQN


def main(args_input):
    agent_classes = {
        'AgentDDPG': AgentDDPG,
        'AgentPPO': AgentPPO,
        'AgentSAC': AgentSAC,
        'AgentTD3': AgentTD3,
        'AgentPDDPG': AgentPDDPG,
        'AgentPSAC': AgentPSAC,
        'AgentPTD3': AgentPTD3,
        'AgentDDPG_single': AgentDDPG_single,
        'AgentDQN': AgentDQN,
        'AgentMADDPG': AgentMADDPG,
        'AgentMASAC': AgentMASAC,
        'AgentMATD3': AgentMATD3,
        'AgentMADQN': AgentMADQN,
        'AgentMADDQN': AgentMADDQN,
    }
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_script_path)
    node_count = args_input.node_count
    env_config['network_info']['bus_info_file'] = os.path.join(parent_dir,
                                                               'rl_adn/data_sources/network_data/node_{}/Nodes_{}.csv'.format(
                                                                   node_count, node_count))
    env_config['network_info']['branch_info_file'] = os.path.join(parent_dir,
                                                                  'rl_adn/data_sources/network_data/node_{}/Lines_{}.csv'.format(
                                                                      node_count, node_count))
    env_config['network_info']['time_series_data_path'] = os.path.join(parent_dir,
                                                                       'rl_adn/data_sources/time_series_data/{}_node_time_series.csv'.format(
                                                                           node_count))
    env_config['time_series_data_path'] = os.path.join(parent_dir,
                                                       'rl_adn/data_sources/time_series_data/{}_node_time_series.csv'.format(
                                                           node_count))
    env_config['seed'] = 521
    env_config['long_term_eval_days'] = args_input.long_term_eval_days
    env_config['node_count'] = node_count
    if node_count == 25:
        env_config['battery_list'] = [4, 8, 10, 13, 14, 16, 22, 24]
    elif node_count == 34:
        env_config['battery_list'] = [11, 15, 26, 29, 33]
    data_path = 'node_{}'.format(node_count)
    env_config['Discrete_value'] = np.array([-1, 0, 1], dtype=np.float32)
    env_config['is_split'] = args_input.is_split
    env = PowerNetEnv(env_config)
    env.agent_classes = args_input.agent_class
    env.if_probability_discrete = args_input.if_probability_discrete
    env_args = {
        'env_name': 'PowerNetEnv',
        'state_dim': env.state_space.shape[0],
        'action_dim': env.action_space.shape[0],
        'if_discrete': False,
        'if_probability_discrete': args_input.if_probability_discrete,
        'num_envs': 1,
        'data_path': data_path
    }
    if args_input.is_long_term_eval:
        env_args['env_name'] = 'PowerNetEnv_7days'
    args = Config(agent_class=agent_classes[args_input.agent_class], env_class=None,
                  env_args=env_args)  # see `Config` for explanation
    args.if_probability_discrete = args_input.if_probability_discrete
    args.num_agents = env.action_space.shape[0]
    '''init buffer configuration'''
    args.discrete_value = env_config['Discrete_value']
    args.gamma = 0.99  # discount factor of future rewards
    args.target_step = 1000
    args.warm_up = args_input.warm_up
    args.if_use_per = False
    args.per_alpha = 0.6
    args.per_beta = 0.4
    args.buffer_size = int(4e5)
    args.repeat_times = 1
    args.batch_size = 512

    '''init device'''
    GPU_ID = args_input.GPU_ID
    args.gpu_id = GPU_ID

    args.random_seed = 521

    '''init agent configration'''
    args.net_dims = (256, 256, 256)
    args.learning_rate = 6e-5

    '''init before training'''
    args.run_name = args_input.run_name
    args.num_episode = args_input.num_episode
    env.time_split_num = args_input.time_split_num
    env.sample_num = args_input.sample_num 
    args.train = args_input.train
    args.time_limit = args_input.time_limit

    if args.train:
        args.if_remove = True
        args.init_before_training()
        log_file = os.path.join(args.cwd, "training_log.txt")

        logging.basicConfig(
            filename=log_file,
            filemode='a',  # 追加模式
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
    else:
        args.if_remove = False
        args.init_before_training()
    env.log_dir = args.cwd
    args.print()
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, args.discrete_value, gpu_id=args.gpu_id,
                             args=args)
    print(agent.device)
    '''init buffer '''
    buffers = [ReplayBuffer(
        gpu_id=args.gpu_id,
        num_seqs=1,
        max_size=args.buffer_size,
        state_dim=args.state_dim,
        action_dim=1,
        if_use_per=args.if_use_per,
        args=args,
    ) for _ in range(args.num_agents)]
    if args.if_off_policy:
        # if_random = True means random action
        buffer_items = agent.explore_env(env, args.target_step, if_random=True)
        for i in range(args.num_agents):
            current_buffer_items = (
                buffer_items[0][:, i:i + 1, :],
                buffer_items[1][:, i:i + 1, :],
                buffer_items[2][:, i:i + 1],
                buffer_items[3][:, i:i + 1]
            )
            buffers[i].update(current_buffer_items)  # warm up for ReplayBuffer

    '''train loop'''
    if args.train:
        time_limit = args.time_limit * 3600
        start_time_laurent = time.time()
        collect_data = True
        while collect_data:
            print(f'buffer:{buffers[0].cur_size}')
            with torch.no_grad():
                buffer_items = agent.explore_env(env, args.target_step, if_random=True)
                for i in range(args.num_agents):
                    current_buffer_items = (
                        buffer_items[0][:, i:i + 1, :],
                        buffer_items[1][:, i:i + 1, :],
                        buffer_items[2][:, i:i + 1],
                        buffer_items[3][:, i:i + 1]
                    )
                    buffers[i].update(current_buffer_items)
            if buffers[0].cur_size >= args.warm_up:
                collect_data = False
        torch.set_grad_enabled(False)

        for i_episode in range(args.num_episode):
            elapsed_time = time.time() - start_time_laurent
            if elapsed_time > time_limit:
                print(f"Time limit of {time_limit / 3600} hours reached. Stopping training.")
                break
            torch.set_grad_enabled(True)
            agent.update_net(buffers)
            torch.set_grad_enabled(False)
            env.train = 1
            (episode_reward, violation_time, violation_value, reward_for_power, reward_for_good_action,
             reward_for_penalty, action_list) = get_episode_return(env, agent, agent.device)
            log_message = (f'curren epsiode is {i_episode}, reward: {episode_reward},'
                           f'violation time of one day for all nodes: {violation_time},'
                           f'violation value is {violation_value}, buffer_length: {buffers[0].cur_size}')
            print(log_message)
            logging.info(log_message)  # 写入日志文件

            if i_episode % 1 == 0:
                buffer_items = agent.explore_env(env, args.target_step, if_random=False)
                for i in range(args.num_agents):
                    current_buffer_items = (
                        buffer_items[0][:, i:i + 1, :],
                        buffer_items[1][:, i:i + 1, :],
                        buffer_items[2][:, i:i + 1],
                        buffer_items[3][:, i:i + 1]
                    )
                    buffers[i].update(current_buffer_items)  # warm up for ReplayBuffer

        agent_name = args.agent_class.__name__[5:]
        time_laurent = time.time() - start_time_laurent
        print(f'time laurent for {agent_name} is {time_laurent}')
        for i in range(args.num_agents):
            agent_dir = os.path.join(args.cwd, f"{i}_agent")
            os.makedirs(agent_dir, exist_ok=True)
            agent.agents[i].save_or_load_agent(agent_dir, if_save=True)
        print('actor and critic parameters have been saved')
        print('training finished')
    else:
        actions_file = os.path.join(args.cwd, 'instances_actions_longterm.txt')
        fitness_file = os.path.join(args.cwd, 'fitness_longterm.txt')
        action_file = os.path.join(args.cwd, 'actions.txt')
        env.actions_file = actions_file
        env.fitness_file = fitness_file
        env.action_file = action_file
        if os.path.exists(actions_file):
            os.remove(actions_file)
        if os.path.exists(fitness_file):
            os.remove(fitness_file)
        if os.path.exists(action_file):
            os.remove(action_file)

        for i in range(args.num_agents):
            agent_dir = os.path.join(args.cwd, f"{i}_agent")
            os.makedirs(agent_dir, exist_ok=True)
            agent.agents[i].save_or_load_agent(agent_dir, if_save=False)
        env.train = -1
        env.sample_num = env.test_day_num
        (episode_reward, violation_time, violation_value, reward_for_power, reward_for_good_action,
         reward_for_penalty) = get_test_return(env, agent, agent.device)
        print(f'Test reward: {episode_reward},'
              f'violation time of one day for all nodes: {violation_time},'
              f'violation value is {violation_value}')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--node_count', type=int, default=34)
    parser.add_argument('--agent_class', type=str, default='AgentSAC', help='Agent class name')
    parser.add_argument('--run_name', type=str, default='ADN-ESS-Remote-2-19-8', help='Name of the run')
    parser.add_argument('--num_episode', type=int, default=5)
    parser.add_argument('--warm_up', type=int, default=2000)
    parser.add_argument('--train', type=str2bool, default=False)
    parser.add_argument('--time_split_num', type=int, default=96)
    parser.add_argument('--sample_num', type=int, default=10)
    parser.add_argument('--GPU_ID', type=int, default=0)
    parser.add_argument('--time_limit', type=int, default=8)
    parser.add_argument('--long_term_eval_days', type=int, default=1)
    parser.add_argument('--if_probability_discrete', type=str2bool, default=False)
    parser.add_argument('--is_split', type=str2bool, default=False)
    parser.add_argument('--is_long_term_eval', type=str2bool, default=False)
    return parser.parse_args()


def print_args(args):
    print("\nParsed arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")


if __name__ == '__main__':
    agent_classs = ['AgentMADDPG', 'AgentMASAC', 'AgentMATD3']
    agent_classs = ['AgentMATD3']
    run_names = ['ADN-ESS-short-multi-agent-48h']
    node_acounts = [25, 34]
    node_acounts = [25]
    for agent_class in agent_classs:
        for i, run_name in enumerate(run_names):
            for node_acount in node_acounts:
                args = parse_args()
                args.time_limit = 16
                args.agent_class = agent_class
                args.long_term_eval_days = 1
                args.train = False
                args.is_split = False
                args.is_long_term_eval = False
                args.run_name = run_name
                args.node_count = node_acount
                print_args(args)
                print(f"Agent: {agent_class}, Run name: {args.run_name}")  # 输出运行名称
                main(args)
