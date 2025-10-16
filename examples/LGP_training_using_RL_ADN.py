import argparse
import copy
import operator
import random
import time
import os
import pickle
import shutil
import sys

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from multiprocessing import Pool
import multiprocessing

from rl_adn.environments.env import PowerNetEnv, env_config
from rl_adn.DRL_algorithms.Agent import AgentDDPG
from rl_adn.DRL_algorithms.utility import Config, ReplayBuffer, get_episode_return
from concurrent.futures import ProcessPoolExecutor
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import pandas as pd
import importlib
importlib.reload(tools)
from operator import attrgetter

'''Define new functions'''
SEED = 521
random.seed(SEED)
np.random.seed(SEED)


def Div(left, right):
    return left / right if right != 0 else 1


def init_pool():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def random_fun():
    return np.random.rand()


def process(funcs, env):
    time_split_num = env.time_split_num * env.long_term_eval_days
    state = env.reset()
    episode_return = 0.0
    combine_feature = np.empty((len(env.operator), state.size + 1))
    rewards = np.zeros(time_split_num)
    actions_buffer = np.zeros((time_split_num, env.action_dim))

    for step in range(time_split_num):
        origin_feature = state
        op_values = env.operator
        combine_feature[:, :-1] = origin_feature
        combine_feature[:, -1] = env.operator

        action_values = funcs.evaluate(combine_feature)
        actions = op_values[np.argmin(action_values, axis=0)]

        actions_buffer[step] = actions
        next_state, reward, done, _ = env.step(actions)
        rewards[step] = reward
        episode_return += reward
        state = next_state
        if done:
            break

    if env.train == -1:
        with open(env.actions_file, 'a') as f:
            Instance_name = env.data_manager.test_dates[env.idx]
            f.write(f"Instances {env.idx}: {Instance_name}\n")
            for step_actions in actions_buffer:
                actions_str = ' '.join(map(str, step_actions))
                f.write(f"{actions_str}\n")
            f.write(f"Fitness: {episode_return}\n")
        with open(env.fitness_file, 'a') as f:
            f.write(f"{episode_return}\n")
    return episode_return


def evaluate(individual, env, toolbox):
    env.train = 1
    sample_nums = env.sample_num

    funcs = individual
    all_rewards = []
    for _ in range(sample_nums):
        rewards = process(funcs, env)
        all_rewards.append(rewards)
    return (np.mean(all_rewards),)

def evaluate_test(individual, env, toolbox):
    sample_nums = env.test_day_num
    env.train = -1
    all_rewards = np.zeros(sample_nums)

    log_dir = env.log_dir
    actions_file = os.path.join(log_dir, 'instances_actions_longterm.txt')
    fitness_file = os.path.join(log_dir, 'fitness_longterm.txt')
    action_file = os.path.join(log_dir, 'actions.txt')
    env.actions_file = actions_file
    env.fitness_file = fitness_file
    env.action_file = action_file
    if os.path.exists(actions_file):
        os.remove(actions_file)
    if os.path.exists(fitness_file):
        os.remove(fitness_file)
    if os.path.exists(action_file):
        os.remove(action_file)

    funcs = individual
    for idx in range(sample_nums):
        env.idx = idx
        day_cost = process(funcs, env)
        all_rewards[idx] = day_cost
        print(f'{idx}/{sample_nums} day finish !!!')
    return (all_rewards.mean(), all_rewards.std())


def evaluate_individual(args):
    individual, toolbox, env, train_day_num = args
    funcs = individual
    day_costs = np.zeros(train_day_num)
    for idx in range(train_day_num):
        env.idx = idx
        day_costs[idx] = process(funcs, env)
    return np.mean(day_costs)


def init_worker():
    pid = os.getpid()
    urand = int.from_bytes(os.urandom(4), byteorder='big')
    seed = (urand + pid) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def find_best_population_parallel(env, population, toolbox, max_workers):
    env.train = 0
    train_day_num = env.train_day_num
    args = [(copy.deepcopy(individual), toolbox, env, train_day_num) for individual in population]
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        individual_means = list(executor.map(evaluate_individual, args))

    row_means = np.array(individual_means)
    max_index = np.argmax(row_means)
    return population[max_index]


class Instruction:
    def __init__(self, name, args, result):
        self.name = name
        self.args = args
        self.result = result

    def __str__(self):
        return self.name


instructions = [
    Instruction("add", ['register', 'register'], 'register'),
    Instruction("sub", ['register', 'register'], 'register'),
    Instruction("mul", ['register', 'register'], 'register'),
    Instruction("div", ['register', 'register'], 'register'),
    Instruction("max", ['register', 'register'], 'register'),
    Instruction("min", ['register', 'register'], 'register'),
    Instruction("neg", ['register'], 'register'),
    Instruction("mov_r", ['register'], 'register'),
]


def make_constants(output_dim):
    return [random.uniform(-1, 1) for _ in range(output_dim)]


class MultiInputLGPIndividual(list):
    def __init__(self, program_length=100, input_dim=52, constants_num=4, output_dim=8, out_regis_num=16):
        super().__init__()
        self.program_length = program_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.out_regis_num = out_regis_num
        self.constants_num = constants_num
        self.constants = make_constants(constants_num)
        self.regis_num = input_dim + constants_num + output_dim + out_regis_num

        for _ in range(program_length):
            op = random.choice(instructions)
            args = []

            # 选择操作数
            for arg_type in op.args:
                if arg_type == 'register':
                    args.append(random.randint(0, self.regis_num - 1))
            if op.result == 'register':
                out = random.randint(input_dim + constants_num, self.regis_num - 1)
            self.append((op, args, out))

    def reset(self, data, instruction_map):
        self.clear()
        self.constants = data["constants"][:]
        for name, args, out in data["program"]:
            op = instruction_map[name]
            self.append((op, args, out))

    def evaluate(self, inputs):
        m, n = inputs.shape
        registers = np.ones((m, self.regis_num), dtype=np.float64)
        const_tiled = np.tile(self.constants, (m, 1))
        registers[:, :self.input_dim] = inputs
        registers[:, self.input_dim:self.input_dim + self.constants_num] = const_tiled

        for op, arg_indices, out in self:
            if op.name in ('add', 'sub', 'mul', 'div', 'max', 'min'):
                arg1_val = self._get_arg_value(op.args[0], arg_indices[0], inputs, registers, const_tiled, m)
                arg2_val = self._get_arg_value(op.args[1], arg_indices[1], inputs, registers, const_tiled, m)
                if op.name == 'add':
                    val = arg1_val + arg2_val
                elif op.name == 'sub':
                    val = arg1_val - arg2_val
                elif op.name == 'mul':
                    val = arg1_val * arg2_val
                elif op.name == 'div':
                    mask = np.abs(arg2_val) > 1e-6
                    val = np.ones_like(arg1_val, dtype=float)
                    val[mask] = arg1_val[mask] / arg2_val[mask]
                elif op.name == 'max':
                    val = np.maximum(arg1_val, arg2_val)
                else:
                    val = np.minimum(arg1_val, arg2_val)
            else:
                arg1_val = self._get_arg_value(op.args[0], arg_indices[0], inputs, registers, const_tiled, m)
                if op.name == 'neg':
                    val = -arg1_val
                else:
                    val = arg1_val
            registers[:, out] = val

        return registers[:, -self.output_dim:]

    def _get_arg_value(self, arg_type, idx, inputs, registers, const_tiled, m):
        if arg_type == 'input':
            return inputs[:, idx]
        elif arg_type == 'register':
            return registers[:, idx]
        elif arg_type == 'constant':
            return const_tiled[:, idx]
        else:
            return 0.0


def cx_individual(ind1, ind2):
    size = min(ind1.program_length, ind2.program_length)
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    return ind1, ind2


def mutate_individual_refined(individual):
    input_dim = individual.input_dim
    constants_num = individual.constants_num
    regis_num = individual.regis_num
    mutation_num = int(individual.program_length * 0.1)
    indices = random.sample(range(0, individual.program_length), mutation_num)

    for i in indices:
        mutation_type = random.choice(['op', 'arg', 'dest'])
        op, args, out = individual[i]

        if mutation_type == 'op':
            new_op = random.choice(instructions)
            if len(new_op.args) != len(op.args):
                new_args = []
                for _ in new_op.args:
                    new_args.append(random.randint(0, regis_num - 1))
                individual[i] = (new_op, new_args, out)
            else:
                individual[i] = (new_op, args, out)

        elif mutation_type == 'dest':
            new_out = random.randint(input_dim + constants_num, regis_num - 1)
            individual[i] = (op, args, new_out)

        elif mutation_type == 'arg' and args:
            arg_idx_to_mutate = random.randint(0, len(args) - 1)
            new_args = list(args)
            new_args[arg_idx_to_mutate] = random.randint(0, regis_num - 1)
            individual[i] = (op, new_args, out)
    return individual,


def main(args_input):
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_script_path)
    '''Load dataset'''
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

    env_config['seed'] = 512
    env_config['long_term_eval_days'] = args_input.long_term_eval_days
    env_config['node_count'] = node_count
    env_config['Discrete_value'] = np.array([-1, 0, 1])
    if node_count == 25:
        env_config['battery_list'] = [4, 8, 10, 13, 14, 16, 22, 24]
    elif node_count == 34:
        env_config['battery_list'] = [11, 15, 26, 29, 33]
    data_path = 'node_{}'.format(node_count)
    env_config['is_split'] = args_input.is_split
    env = PowerNetEnv(env_config)

    env_args = {
        'env_name': 'PowerNetEnv',
        'state_dim': env.state_space.shape[0],
        'action_dim': env.action_space.shape[0],
        'if_discrete': False,
        'data_path': data_path
    }
    args = Config(agent_class=AgentDDPG, env_class=None, env_args=env_args)
    args.random_seed = 521

    '''init before training'''
    args.agent_name = args_input.agent_class
    args.run_name = args_input.run_name
    args.num_episode = args_input.num_episode
    args.time_limit = args_input.time_limit
    args.population = args_input.population_num

    env.population = args.population
    env.time_split_num = args_input.time_split_num
    env.sample_num = args_input.sample_num
    env.operator_num = 3
    env.operator = np.array([-1, 0, 1])
    env.action_dim = env_args['action_dim']

    args.train = args_input.train
    if args.train:
        args.if_remove = True
        args.cwd = f"./{env_args['env_name']}/{data_path}/{args_input.agent_class}/{args_input.run_name}"
        args.init_before_training()
        args.cwd = f"./{env_args['env_name']}/{data_path}/{args_input.agent_class}/{args_input.run_name}"
    else:
        args.if_remove = False
        args.cwd = f"./{env_args['env_name']}/{data_path}/{args_input.agent_class}/{args_input.run_name}"
        args.init_before_training()
        args.cwd = f"./{env_args['env_name']}/{data_path}/{args_input.agent_class}/{args_input.run_name}"
    args.print()

    feature_num = env.state_length + 1

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", MultiInputLGPIndividual, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", creator.Individual, program_length=args_input.program_length,
                     input_dim=feature_num, output_dim=env.action_dim, out_regis_num=args_input.regis_num)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", cx_individual)
    toolbox.register("mutate", mutate_individual_refined)

    log_dir = args.cwd
    env.log_dir = log_dir
    max_workers = args_input.max_workers
    max_height = 8
    if args.train:
        time_limit = args.time_limit * 3600
        start_time_laurent = time.time()

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        os.makedirs(log_dir, exist_ok=True)
        population = toolbox.population(n=args.population)

        random.seed(args.random_seed)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        population, log = algorithms.eaWithBest_parallel(
            population, toolbox, cxpb=0.7, mutpb=0.2, repb=0.1, ngen=args.num_episode, time_limit=time_limit, env=env,
            pset=None,
            max_workers=max_workers, max_height=max_height, stats=stats, start_time=start_time_laurent, verbose=True
        )

        log_df = pd.DataFrame(log)
        csv_file = os.path.join(log_dir, "gp_log.csv")
        log_df.to_csv(csv_file, index=False)

        best_individuals_file = os.path.join(log_dir, "best_individuals.pkl")
        best_individual = find_best_population_parallel(env, population, toolbox, max_workers)
        func = best_individual
        with open(os.path.join(log_dir, "best_individuals.txt"), "w") as f:
            f.write(f"Rank 1: {func}\n"
                    f"Fitness: {best_individual.fitness.values}\n\n")
        print("Best Individual:", best_individual)
        print("Fitness:", best_individual.fitness.values)

        serializable_ind = {
            "program": [(op.name, args, out) for op, args, out in best_individual],
            "program_length": best_individual.program_length,
            "constants_num": best_individual.constants_num,
            "constants": best_individual.constants,
            "input_dim": best_individual.input_dim,
            "output_dim": best_individual.output_dim,
            "out_regis_num": best_individual.out_regis_num,
            "regis_num": best_individual.regis_num,
        }
        with open(best_individuals_file, "wb") as f:
            pickle.dump(serializable_ind, f)
    else:
        loaded_best_individuals_file = os.path.join(log_dir, "best_individuals.pkl")

        with open(loaded_best_individuals_file, "rb") as f:
            data = pickle.load(f)

        instruction_map = {instr.name: instr for instr in instructions}

        loaded_best_individual = MultiInputLGPIndividual(
            program_length=data["program_length"],
            input_dim=data["input_dim"],
            output_dim=data["output_dim"],
            constants_num=data["constants_num"],
            out_regis_num=data["out_regis_num"]
        )

        loaded_best_individual.reset(data, instruction_map)
        mean_reward = evaluate_test(loaded_best_individual, env, toolbox)
        print(f"Test reward: {mean_reward}\n")


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
    parser.add_argument('--agent_class', type=str, default='LGP', help='Agent class name')
    parser.add_argument('--run_name', type=str, default='ADN-ESS-Remote-TEST', help='Name of the run')
    parser.add_argument('--num_episode', type=int, default=2000)
    parser.add_argument('--population_num', type=int, default=8)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--time_split_num', type=int, default=96)
    parser.add_argument('--sample_num', type=int, default=20)
    parser.add_argument('--GPU_ID', type=int, default=0)
    parser.add_argument('--time_limit', type=int, default=800000)
    parser.add_argument('--long_term_eval_days', type=int, default=1)
    parser.add_argument('--isEncoding', type=str2bool, default=False)
    parser.add_argument('--max_workers', type=int, default=10)
    parser.add_argument('--is_split', type=str2bool, default=False)
    parser.add_argument('--is_long_term_eval', type=str2bool, default=False)
    parser.add_argument('--program_length', type=int, default=100)
    parser.add_argument('--regis_num', type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    run_names = ['ADN-EES-short-48h']
    node_counts = [25, 34]
    node_counts = [34]
    for run_name in run_names:
        for node_count in node_counts:
            args = parse_args()
            args.train = False
            args.isEncoding = False
            args.is_long_term_eval = False
            args.long_term_eval_days = 1
            args.run_name = run_name
            args.node_count = node_count
            print(f"Run name: {args.run_name}")
            main(args)
