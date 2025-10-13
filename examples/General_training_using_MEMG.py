import time
import logging
import os
import numpy as np
import argparse
from rl_adn.environments.env import MEMG_config, IESEnv
import torch
from rl_adn.DRL_algorithms.DDPG import AgentDDPG
from rl_adn.DRL_algorithms.PPO import AgentPPO
from rl_adn.DRL_algorithms.SAC import AgentSAC
from rl_adn.DRL_algorithms.TD3 import AgentTD3
from rl_adn.DRL_algorithms.DQN import AgentDQN
from rl_adn.DRL_algorithms.P_DDPG import AgentPDDPG
from rl_adn.DRL_algorithms.P_SAC import AgentPSAC
from rl_adn.DRL_algorithms.P_TD3 import AgentPTD3
from rl_adn.DRL_algorithms.DeRL_SAC import AgentDeRL_SAC
from rl_adn.DRL_algorithms.utility import Config, ReplayBuffer,get_episode_return_MEMG, get_test_return_MEMG


def main(args_input):
    agent_classes = {
        'AgentDDPG': AgentDDPG,
        'AgentPPO': AgentPPO,
        'AgentSAC': AgentSAC,
        'AgentTD3': AgentTD3,
        'AgentPDDPG': AgentPDDPG,
        'AgentPSAC': AgentPSAC,
        'AgentPTD3': AgentPTD3,
        'AgentDQN': AgentDQN,
        'AgentDeRL_SAC': AgentDeRL_SAC,
    }
    MEMG_config['seed'] = 521
    MEMG_config['long_term_eval_days'] = args_input.long_term_eval_days
    MEMG_config['train'] = True
    MEMG_config['intervel'] = args_input.time_split_num
    MEMG_config['MEMG_data'] = os.path.join('../data_sources/MEMG', args_input.data_path)
    MEMG_config['Discrete_value'] = np.array([-1, 0, 1])
    MEMG_config['is_split'] = args_input.is_split
    env = IESEnv(MEMG_config)
    env.if_probability_discrete = args_input.if_probability_discrete
    env_args = {
        'env_name': 'MEMG',
        'state_dim': env.state_space.shape[0],
        'action_dim': env.action_space.shape[0],
        'if_discrete': False,
        'if_probability_discrete': args_input.if_probability_discrete,
        'num_envs': 1,
        'max_step': 24,
        'data_path': args_input.data_path[:-4]
    }
    if args_input.is_long_term_eval:
        env_args['env_name'] = 'MEMG_7days'
    env.agent_classes = args_input.agent_class

    args = Config(agent_class=agent_classes[args_input.agent_class], env_class=None,
                  env_args=env_args)
    args.if_probability_discrete = args_input.if_probability_discrete

    '''init buffer configuration'''
    args.discrete_value = MEMG_config['Discrete_value']
    args.gamma = 0.99
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
    args.num_workers = 4
    args.num_threads = 4
    args.random_seed = 521

    '''init agent configration'''
    args.net_dims = (256, 256, 256)
    args.learning_rate = 6e-4
    args.learning_rate = 1e-3

    '''init before training'''
    args.run_name = args_input.run_name
    args.num_episode = args_input.num_episode
    args.time_limit = args_input.time_limit
    env.time_split_num = args_input.time_split_num
    env.sample_num = args_input.sample_num
    args.train = args_input.train
    if args.train:
        args.if_remove = True
        args.init_before_training()
        log_file = os.path.join(args.cwd, "training_log.txt")
        logging.basicConfig(
            filename=log_file,
            filemode='a',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
    else:
        args.if_remove = False
        args.init_before_training()
    args.print()

    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, args.discrete_value, gpu_id=args.gpu_id,
                             args=args)
    print(agent.device)
    '''init buffer '''
    buffer = ReplayBuffer(
        gpu_id=args.gpu_id,
        num_seqs=args.num_envs,
        max_size=args.buffer_size,
        state_dim=args.state_dim,
        action_dim=1 if args.if_discrete else args.action_dim,
        if_use_per=args.if_use_per,
        args=args,
    )
    if args.if_off_policy:
        buffer_items = agent.explore_env(env, args.target_step, if_random=True)
        buffer.update(buffer_items)

    '''train loop'''
    if args.train:
        time_limit = args.time_limit * 3600
        start_time_laurent = time.time()
        collect_data = True
        while collect_data:
            print(f'buffer:{buffer.cur_size}')
            with torch.no_grad():
                buffer_items = agent.explore_env(env, args.target_step, if_random=True)
                buffer.update(buffer_items)
            if buffer.cur_size >= args.warm_up:
                collect_data = False
        torch.set_grad_enabled(False)

        for i_episode in range(args.num_episode):
            elapsed_time = time.time() - start_time_laurent
            if elapsed_time > time_limit:
                print(f"Time limit of {time_limit / 3600} hours reached. Stopping training.")
                break
            torch.set_grad_enabled(True)
            agent.update_net(buffer)
            torch.set_grad_enabled(False)
            env.train = 1
            (episode_reward, violation_time, violation_value, reward_for_power, reward_for_good_action,
             reward_for_penalty, action_list) = get_episode_return_MEMG(env, agent, agent.device)
            log_message = (f'curren epsiode is {i_episode}, reward: {episode_reward},'
                           f'violation time of one day for all nodes: {violation_time},'
                           f'violation value is {violation_value}, buffer_length: {buffer.cur_size}')
            print(log_message)
            logging.info(log_message)

            if i_episode % 1 == 0:
                buffer_items = agent.explore_env(env, args.target_step, if_random=False)
                buffer.update(buffer_items)

        agent_name = args.agent_class.__name__[5:]
        time_laurent = time.time() - start_time_laurent
        print(f'time laurent for {agent_name} is {time_laurent}')
        agent.save_or_load_agent(args.cwd, if_save=True)
        buffer.save_or_load_history(args.cwd, if_save=True)
        print('actor and critic parameters have been saved')
        print('training finished')
    else:
        info_file = os.path.join(args.cwd, 'infor_longterm.txt')
        fitness_file = os.path.join(args.cwd, 'fitness_longterm.txt')
        action_file = os.path.join(args.cwd, 'action_longterm.txt')
        env.info_file = info_file
        env.fitness = fitness_file
        env.action_file = action_file

        if os.path.exists(info_file):
            os.remove(info_file)
        if os.path.exists(fitness_file):
            os.remove(fitness_file)
        if os.path.exists(action_file):
            os.remove(action_file)

        agent.save_or_load_agent(args.cwd, if_save=False)
        env.sample_num = env.test_day_num
        env.train = -1
        (episode_reward, std_reward, violation_value, reward_for_power, reward_for_good_action,
         reward_for_penalty, action_list) = get_test_return_MEMG(env, agent, agent.device)
        print(f'Avg reward: {episode_reward}, Std: {std_reward}')


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
    parser.add_argument('--agent_class', type=str, default='AgentDDPG', help='Agent class name')
    parser.add_argument('--run_name', type=str, default='MEMG-before2-Remote-2-19-8', help='Name of the run')
    parser.add_argument('--num_episode', type=int, default=99999999)
    parser.add_argument('--warm_up', type=int, default=2000)
    parser.add_argument('--train', type=str2bool, default=False)
    parser.add_argument('--time_split_num', type=int, default=24)
    parser.add_argument('--sample_num', type=int, default=20)
    parser.add_argument('--GPU_ID', type=int, default=0)
    parser.add_argument('--time_limit', type=int, default=12)
    parser.add_argument('--data_path', type=str, default='concate_year.csv')
    parser.add_argument('--if_probability_discrete', type=str2bool, default=False)
    parser.add_argument('--long_term_eval_days', type=int, default=1)
    parser.add_argument('--is_split', type=str2bool, default=False)
    parser.add_argument('--is_long_term_eval', type=str2bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    agent_classs = ['AgentDDPG', 'AgentSAC', 'AgentTD3']
    agent_classs = ['AgentTD3']
    run_names = ['MEMG-Remote-continue-12h']
    data_paths = ['concate_year.csv', 'american_data.csv', 'bilishi_data.csv']
    data_paths = ['american_data.csv']
    for agent_class in agent_classs:
        for i, run_name in enumerate(run_names):
            for data_path in data_paths:
                args = parse_args()
                args.agent_class = agent_class
                args.train = False
                args.is_split = False
                args.is_long_term_eval = False
                args.long_term_eval_days = 1
                args.run_name = run_name
                args.data_path = data_path
                print(f"Agent: {agent_class}, Run name: {args.run_name}")  # 输出运行名称
                main(args)