import copy as cp
import random
import warnings

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandapower as pp
import pandas as pd

from rl_adn.data_manager.data_manager import GeneralPowerDataManager, MEMG_train_and_test_data
from rl_adn.environments.battery import Battery, battery_parameters
from rl_adn.utility.grid import GridTensor
from rl_adn.utility.utils import create_pandapower_net
from scipy import integrate

env_config = {"voltage_limits": [0.95, 1.05],
              "algorithm": "Laurent",
              "battery_list": [11, 15, 26, 29, 33],
              "year": 2020, "month": 1, "day": 1,
              "train": True,
              "state_pattern": "default",
              "network_info":
                  {'vm_pu': 1.0, 's_base': 1000,
                   'bus_info_file': '../data_sources/network_data/node_34/Nodes_34.csv',
                   'branch_info_file': '../data_sources/network_data/node_34/Lines_34.csv'},
              "time_series_data_path": "../data_sources/time_series_data/34_node_time_series.csv"}

MEMG_config = {'num_episodes': 20000,
                'batch_size': 256,
                "train": True,
                # days range
                'DEFAULT_DAY0': 0,
                # Length of one episode
                'DEFAULT_ITERATIONS': 24,  # hours

                'SOLAR_OPERATION_COST': 0.02,  # $/KW
                'GB_OPERATION_COST': 0.2,  # $/KW CHP运行成本与燃料成本整合在一起
                'ES_OPERATION_COST': 0.1,  # $/KW

                # Prices $/KW
                'DEFAULT_ELEC_MARKET_PRICE': np.array(
                    [0.2652, 0.2580, 0.2547, 0.2528, 0.2517, 0.2543, 0.2857, 0.3238, 0.3399, 0.3452, 0.3397, 0.3364,
                     0.3339, 0.3257, 0.3292, 0.3211, 0.3217, 0.3297, 0.3474, 0.3424, 0.3232, 0.3261, 0.2993, 0.2831]),
                'DEFAULT_HEAT_MARKET_PRICE': 0.48,  # $/KW
                'DEFAULT_GAS_MARKET_PRICE': 0.4,  # $/Sm3

                # Constraints
                'CHP_MIN_POWER': 0,
                'CHP_MAX_POWER': 200,  # KW  不仅满足电负荷，还有HP也需要供热
                'GB_MIN_POWER': 0,
                'GB_MAX_POWER': 300,  # KW

                # EES capacity
                'DEFAULT_EES_CAPACITY': 200,  # KW
                'DEFAULT_MIN_EES_SOC': 0.2,
                'DEFAULT_MAX_EES_SOC': 1.0,
                'DEFAULT_MAX_EES_CHARGE': 20,  # 单位时间最大充放电约束

                # 这里设置的和EES一样
                'DEFAULT_TES_CAPACITY': 200,  # KW
                'DEFAULT_MIN_TES_SOC': 0.2,
                'DEFAULT_MAX_TES_SOC': 1.0,
                'DEFAULT_MAX_TES_CHARGE': 20,  # KW

                # COEFFICIENTS
                'YITA_CHARGE_E': 0.95,
                'YITA_DISCHARGE_E': 0.95,
                'YITA_CHARGE_T': 0.95,
                'YITA_DISCHARGE_T': 0.95,

                'ALPHA_GB': 0.8,
                'ALPHA_CHP': 1.2,
                'CHP_a': 0.176,  # $/h
                'CHP_b': 0.135,  # $/KWh
                'CHP_c': 0.000004,  # $/KWh2

                # 奖励函数系数
                'K0': 0.1,  # 运行成本
                'K1': 0.1,  # 燃料成本系数 发电成本
                'K2': 15,  # 碳交易成本
                'K3': 0.5,  # 爬坡约束惩罚
                'K4': 0.5,  # 能源平衡惩罚
                'K5': 100,  # 绿证参数

                # 本project只有CHP、GB产生碳排放
                'INTERVAL': 3,  # 单位：t
                'CARBON_BASE_PRICE': 4,  # $/t
                'BASE_SLOP': 0.5,  # $
                'REWARD_COEFFICIENT': 0.3,
                'CARBON_UNIT_CHP': 0.0048,  # t/KW  根据参考文献的比例设定，CHP的碳配额要比碳排放高
                'CARBON_UNIT_GB': 0.0056,
                # 单位能耗碳排放量，碳配额
                'CARBON_UNIT_GRID': 0.0072,  # t/KW 假设大电网发电全部来自化石燃料
                'CARBON_UNIT_HEAT': 0.0084,  # t/KW 假设热网供热也全部来自化石燃料
                # CHP、GB机组才有配额
                'QUOTE_GB': 0.006,  # t/kw
                'QUOTE_CHP': 0.0072,  # t/kw
                # 绿证模型
                'delta_g': 20,  # $/MW 考虑奖惩阶梯型碳交易和电–热转移负荷不确定性的综合能源系统规划
                'QUOET_EFFI': 0.15,  # 绿证配额系数，1个绿色证书等于1MWh绿电 0.15
                'MEMG_data': '../data_sources/MEMG-before2/concate_year.csv',}
MEMG_config['CHP_RAMP_LIMIT'] = 0.3 * MEMG_config['CHP_MAX_POWER']
MEMG_config['GB_RAMP_LIMIT'] = 0.3 * MEMG_config['GB_MAX_POWER']
MEMG_config['DEFAULT_MAX_EES_DISCHARGE'] = MEMG_config['DEFAULT_MAX_EES_CHARGE']
MEMG_config['DEFAULT_MAX_TES_DISCHARGE'] = MEMG_config['DEFAULT_MAX_TES_CHARGE']


class IESEnv(gym.Env):

    def __init__(self, MEMG_config: dict = MEMG_config)->None:
        self.seed = MEMG_config['seed']
        self.train = MEMG_config['train']
        self.is_split = MEMG_config['is_split']
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.long_term_eval_days = MEMG_config['long_term_eval_days']
        self.iterations = MEMG_config['DEFAULT_ITERATIONS']
        self.time_step_length = self.iterations * self.long_term_eval_days

        self.elec_market_price = np.tile(MEMG_config['DEFAULT_ELEC_MARKET_PRICE'], self.long_term_eval_days)
        self.heat_market_price = MEMG_config['DEFAULT_HEAT_MARKET_PRICE']

        self.ees_capacity = MEMG_config['DEFAULT_EES_CAPACITY']
        self.min_ees_soc = MEMG_config['DEFAULT_MIN_EES_SOC']
        self.max_ees_soc = MEMG_config['DEFAULT_MAX_EES_SOC']
        self.max_ees_charge = MEMG_config['DEFAULT_MAX_EES_CHARGE']
        self.max_ees_discharge = MEMG_config['DEFAULT_MAX_EES_DISCHARGE']

        self.tes_capacity = MEMG_config['DEFAULT_TES_CAPACITY']
        self.min_tes_soc = MEMG_config['DEFAULT_MIN_TES_SOC']
        self.max_tes_soc = MEMG_config['DEFAULT_MAX_TES_SOC']
        self.max_tes_charge = MEMG_config['DEFAULT_MAX_TES_CHARGE']
        self.max_tes_discharge = MEMG_config['DEFAULT_MAX_TES_DISCHARGE']
        self.DEFAULT_ELEC_MARKET_PRICE = MEMG_config['DEFAULT_ELEC_MARKET_PRICE']
        self.max_price = np.max(self.DEFAULT_ELEC_MARKET_PRICE)
        self.min_price = np.min(self.DEFAULT_ELEC_MARKET_PRICE)
        self.SOLAR_OPERATION_COST = MEMG_config['SOLAR_OPERATION_COST']
        self.CHP_RAMP_LIMIT = MEMG_config['CHP_RAMP_LIMIT']
        self.GB_RAMP_LIMIT = MEMG_config['GB_RAMP_LIMIT']
        self.Discrete_value = MEMG_config['Discrete_value']


        (self.train_data_elec, self.train_data_heat, self.train_data_solar, self.train_data_date,
         self.test_data_elec, self.test_data_heat, self.test_data_solar, self.test_data_date) = MEMG_train_and_test_data(MEMG_config['MEMG_data'], self.long_term_eval_days)
        self.train_day_num = len(self.train_data_elec) // 24 - self.long_term_eval_days +1
        self.test_day_num = len(self.test_data_elec) // 24 - self.long_term_eval_days +1

        if self.train:
            self.elec_demand, self.heat_demand = self.train_data_elec, self.train_data_heat
            self.solar_generation = Generation_Solar(self.train_data_solar, self.SOLAR_OPERATION_COST)
        else:
            self.elec_demand, self.heat_demand = self.test_data_elec, self.test_data_heat
            self.solar_generation = Generation_Solar(self.test_data_solar, self.SOLAR_OPERATION_COST)

        self.CARBON_UNIT_GRID = MEMG_config['CARBON_UNIT_GRID']
        self.CARBON_UNIT_HEAT = MEMG_config['CARBON_UNIT_HEAT']
        self.grid_elec = Grid_Electric(self.elec_market_price, self.CARBON_UNIT_GRID)
        self.grid_heat = Grid_Heat(self.heat_market_price, self.CARBON_UNIT_HEAT)

        self.YITA_DISCHARGE_E = MEMG_config['YITA_DISCHARGE_E']
        self.YITA_CHARGE_E = MEMG_config['YITA_CHARGE_E']
        self.YITA_DISCHARGE_T = MEMG_config['YITA_DISCHARGE_T']
        self.YITA_CHARGE_T = MEMG_config['YITA_CHARGE_T']
        self.ES_OPERATION_COST = MEMG_config['ES_OPERATION_COST']
        self.ees_battery = ES(capacity=self.ees_capacity, rateD=self.YITA_DISCHARGE_E, rateC=self.YITA_CHARGE_E,
                              max_soc=self.max_ees_soc, min_soc=self.min_ees_soc,
                              ope_effi=self.ES_OPERATION_COST)

        self.tes_battery = ES(capacity=self.tes_capacity, rateD=self.YITA_DISCHARGE_T, rateC=self.YITA_CHARGE_T,
                              max_soc=self.max_tes_soc, min_soc=self.min_tes_soc,
                              ope_effi=self.ES_OPERATION_COST)
        self.ALPHA_GB = MEMG_config['ALPHA_GB']
        self.DEFAULT_GAS_MARKET_PRICE = MEMG_config['DEFAULT_GAS_MARKET_PRICE']
        self.GB_OPERATION_COST = MEMG_config['GB_OPERATION_COST']
        self.QUOTE_GB = MEMG_config['QUOTE_GB']
        self.CARBON_UNIT_CHP = MEMG_config['CARBON_UNIT_CHP']
        self.CARBON_UNIT_GB = MEMG_config['CARBON_UNIT_GB']
        self.GB = GB(self.ALPHA_GB, self.DEFAULT_GAS_MARKET_PRICE, self.GB_OPERATION_COST, self.QUOTE_GB, self.CARBON_UNIT_GB)
        self.ALPHA_CHP = MEMG_config['ALPHA_CHP']
        self.CHP_a = MEMG_config['CHP_a']
        self.CHP_b = MEMG_config['CHP_b']
        self.CHP_c = MEMG_config['CHP_c']
        self.QUOTE_CHP = MEMG_config['QUOTE_CHP']
        self.CHP = CHP(self.ALPHA_CHP, (self.CHP_a, self.CHP_b, self.CHP_c), self.QUOTE_CHP, self.CARBON_UNIT_CHP)
        self.users = User(self.elec_demand, self.heat_demand)
        self.INTERVAL = MEMG_config['INTERVAL']
        self.CARBON_BASE_PRICE = MEMG_config['CARBON_BASE_PRICE']
        self.BASE_SLOP = MEMG_config['BASE_SLOP']
        self.REWARD_COEFFICIENT = MEMG_config['REWARD_COEFFICIENT']
        self.pw_carbon = PW_Carbon(self.INTERVAL, self.CARBON_BASE_PRICE, self.BASE_SLOP, self.REWARD_COEFFICIENT)
        self.K0 = MEMG_config['K0']
        self.K1 = MEMG_config['K1']
        self.K2 = MEMG_config['K2']
        self.K3 = MEMG_config['K3']
        self.K4 = MEMG_config['K4']
        self.K5 = MEMG_config['K5']

        self.delta_g = MEMG_config['delta_g']
        self.QUOET_EFFI = MEMG_config['QUOET_EFFI']
        self.CHP_MIN_POWER = MEMG_config['CHP_MIN_POWER']
        self.CHP_MAX_POWER = MEMG_config['CHP_MAX_POWER']
        self.GB_MIN_POWER = MEMG_config['GB_MIN_POWER']
        self.GB_MAX_POWER = MEMG_config['GB_MAX_POWER']
        self.green_certificate = green_certificate(delta_g=self.delta_g, quote_effi=self.QUOET_EFFI)

        self.action_length = 5
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_length,), dtype=np.float32)
        self.state_length = 7
        self.state_space = spaces.Box(low=0, high=1, shape=(self.state_length,), dtype=np.float32)

        # Ramp limits
        self.chp_prev = 0
        self.gb_prev = 0
        # The current timestep
        self.time_step = 0
        self.day = 0
        self.idx = self.day

        # 日志信息
        self.carbon_emission = 0.
        self.solar_use = 0.

        # 记录一个episode的各类成本
        # -弃热成本 - 各类发电成本 - 运行成本 - 碳交易成本 - 爬坡约束惩罚 - 能源平衡惩罚
        self.carbon_cost = 0.
        self.generation_cost = 0.
        self.operation_cost = 0.
        self.fuel_cost = 0.
        self.grid_cost = 0.
        self.gc_revenue = 0.  # 每个episode的revenue是常数，没必要加入到reward中
        self.ramp_cost = 0.
        self.balance_cost = 0.
        # self.print_interval = 0

    def get_name(self):
        return "IESEnv"

    def _build_state(self):
        """
        Return current state representation as one vector. Normalization for stable training.
        Return:
            state: 1D vector, containing
        """
        socs = np.array([self.ees_battery.SoC, self.tes_battery.SoC])
        # min—max标准化
        price_b = (self.DEFAULT_ELEC_MARKET_PRICE[self.time_step % self.iterations] - self.min_price) / (self.max_price - self.min_price)
        elec_load = self.users.current_electric_demand(self.time_step)
        heat_load = self.users.current_thermal_demand(self.time_step)
        solar_generation = self.solar_generation.current_generation(self.time_step)
        elec_load = (elec_load - self.users.min_electric_demand) / (self.users.max_electric_demand - self.users.min_electric_demand)
        heat_load = (heat_load - self.users.min_thermal_demand) / (self.users.max_thermal_demand - self.users.min_thermal_demand)
        solar_generation = (solar_generation - self.solar_generation.min_power) / (self.solar_generation.max_power - self.solar_generation.min_power)
        time_step = self.time_step / self.iterations
        state = np.concatenate((socs, [price_b, elec_load, heat_load, solar_generation, time_step]))
        return state

    def _build_info(self):
        return {}

    def step(self, action):
        """
        state: (SoC_ees, SoC_tes, elec_load, heat_load, PV, time) 再查阅一下文献
        action: (P_CHP, H_GB, P_ES, P_TS, Renew_cur), all variables are ranged in [0,1]
        reward: -(购电购气购热成本 + 运行成本 + 碳排放成本 - 绿证收益)
        :return: next_state, reward, terminal, info
        """
        # 改为离散动作
        if self.is_split:
        # action = np.where(action < 0, -1, np.where(action > 0, 1, action))
            threshold = 1/3
            conditions = [
                action < -threshold,
                action > threshold
            ]
            action = np.select(conditions, [-1, 1], default=0)

        # Renew_cur 可再生能源使用量
        CHP_MIN_POWER = self.CHP_MIN_POWER
        CHP_MAX_POWER = self.CHP_MAX_POWER
        GB_MIN_POWER = self.GB_MIN_POWER
        GB_MAX_POWER = self.GB_MAX_POWER
        # 一个step内的各类成本
        ope_step = 0.
        fuel_step = 0.
        carbon_step = 0.
        ramp_step = 0.
        rev_step = 0.
        grid_step = 0.

        CHP_RAMP_LIMIT = self.CHP_RAMP_LIMIT
        GB_RAMP_LIMIT = self.GB_RAMP_LIMIT
        emission = 0.
        elec_grid_buy = 0.
        elec_grid_sell = 0.
        heat_grid_sell = 0.  # 热网售电

        # 将动作空间由【-1,1】转为【0,1】
        action_new = (action + 1) / 2
        P_CHP_ratio, H_GB_ratio, P_ES_ratio, P_TS_ratio, Renew_cur = action_new

        '''电和热需求'''
        elec_demand = self.users.current_electric_demand(self.time_step)
        thermal_demand = self.users.current_thermal_demand(self.time_step)

        '''(1) 热电联用机组部分'''
        # P：发电量KW，H：产热量KW
        P_CHP = CHP_MIN_POWER + P_CHP_ratio * (CHP_MAX_POWER - CHP_MIN_POWER)
        # CHP机组在运行时需要遵守爬坡速率限制
        if self.chp_prev != 0:
            delta = P_CHP - self.chp_prev
            if abs(delta) > CHP_RAMP_LIMIT:
                if self.chp_prev > P_CHP:
                    P_CHP = self.chp_prev + np.sign(delta) * CHP_RAMP_LIMIT
            self.ramp_cost += abs(delta)
            ramp_step += abs(delta)
        else:
            if P_CHP > CHP_RAMP_LIMIT:
                P_CHP = CHP_RAMP_LIMIT
            self.ramp_cost += P_CHP
            ramp_step += P_CHP
        self.chp_prev = P_CHP  # 更新t-1时刻的出力
        # CHP 发电时产热量
        H_CHP = self.CHP.heat_generated(P_CHP, )  # KW 产热量
        # CHP 发电成本、运行成本
        self.fuel_cost += self.CHP.base_ope_cost(P_CHP) / 2
        fuel_step += self.CHP.base_ope_cost(P_CHP) / 2
        self.operation_cost += self.CHP.base_ope_cost(P_CHP) / 2
        ope_step += self.CHP.base_ope_cost(P_CHP) / 2

        '''(2) 煤气锅炉部分'''
        H_GB = GB_MIN_POWER + H_GB_ratio * (GB_MAX_POWER - GB_MIN_POWER)
        # 煤气锅炉爬坡功率
        if self.gb_prev == 0:
            if H_GB > GB_RAMP_LIMIT:
                H_GB = GB_RAMP_LIMIT
            self.ramp_cost += H_GB
            ramp_step += H_GB
        else:
            if abs(self.gb_prev - H_GB) > GB_RAMP_LIMIT:
                if self.gb_prev > H_GB:
                    H_GB = self.gb_prev - GB_RAMP_LIMIT
                else:
                    H_GB = self.gb_prev + GB_RAMP_LIMIT
            self.ramp_cost += abs(self.gb_prev - H_GB)
            ramp_step += abs(self.gb_prev - H_GB)
        self.gb_prev = H_GB  # 更新t-1时刻的出力
        # GB 产热成本、运行成本
        self.fuel_cost += self.GB.base_cost(H_GB)
        fuel_step += self.GB.base_cost(H_GB)
        self.operation_cost += self.GB.operation_cost(H_GB)
        ope_step += self.GB.operation_cost(H_GB)

        '''(3) PV系统部分'''
        solar_energy = self.solar_generation.current_generation(self.time_step)
        renewable = solar_energy * Renew_cur
        self.solar_use += renewable
        # 光伏运行成本
        self.operation_cost += self.solar_generation.operation_cost(renewable)
        ope_step += self.solar_generation.operation_cost(renewable)

        '''(4) 电能系统部分'''
        P_ES = (P_ES_ratio * 2 - 1) * self.max_ees_discharge
        P_ES = np.clip(P_ES, -self.max_ees_discharge, self.max_ees_charge)
        # 充放电环节
        if P_ES > 0:  # charging
            P_ES = self.ees_battery.charge(P_ES)
            self.operation_cost += self.ees_battery.operation_cost(P_ES)
            ope_step += self.ees_battery.operation_cost(P_ES)

            if elec_demand + P_ES > P_CHP + renewable:  # 如果需求>负载，那么买电;
                elec_grid_sell = P_ES + elec_demand - P_CHP - renewable
                emission += self.grid_elec.carbon_emission(elec_grid_sell)
                self.carbon_emission += self.grid_elec.carbon_emission(elec_grid_sell)
                grid_step += self.grid_elec.sell(elec_grid_sell, self.time_step)
                self.grid_cost += self.grid_elec.sell(elec_grid_sell, self.time_step)
            else:
                elec_grid_buy = P_CHP + renewable - elec_demand - P_ES  # 如果需求<负载，那么卖电;
                grid_step += self.grid_elec.buy(P_CHP + renewable - elec_demand - P_ES, self.time_step)
                self.grid_cost += self.grid_elec.buy(P_CHP + renewable - elec_demand - P_ES, self.time_step)
        else:  # discharging
            P_ES = -self.ees_battery.discharge(-P_ES)  # 放电为负数
            self.operation_cost += self.ees_battery.operation_cost(-P_ES)
            ope_step += self.ees_battery.operation_cost(-P_ES)

            if -P_ES + P_CHP + renewable < elec_demand:
                elec_grid_sell = elec_demand + P_ES - P_CHP - renewable
                emission += self.grid_elec.carbon_emission(elec_grid_sell)
                self.carbon_emission += self.grid_elec.carbon_emission(elec_grid_sell)
                grid_step += self.grid_elec.sell(elec_grid_sell, self.time_step)
                self.grid_cost += self.grid_elec.sell(elec_grid_sell, self.time_step)
            else:  # 上网
                elec_grid_buy = -P_ES + P_CHP + renewable - elec_demand
                grid_step += self.grid_elec.buy(elec_grid_buy, self.time_step)
                self.grid_cost += self.grid_elec.buy(elec_grid_buy, self.time_step)

        # if elec_demand + elec_grid_buy == -P_ES + P_CHP + renewable + elec_grid_sell:
        #     print("....")

        '''(5) 热能系统部分'''
        H_TS = (P_TS_ratio * 2 - 1) * self.max_tes_charge
        H_TS = np.clip(H_TS, -self.max_tes_discharge, self.max_tes_charge)

        # 弃热成本 暂时不考虑
        H_CHP_GB = H_CHP + H_GB
        if H_TS > 0:  # charging
            H_TS = self.tes_battery.charge(H_TS)  # 获取实际操作值
            ope_step += self.tes_battery.operation_cost(H_TS)  # 计算操作花费
            self.operation_cost += self.tes_battery.operation_cost(H_TS)

            # 多余热量先给TES充电，然后再弃热
            if H_CHP_GB >= thermal_demand + H_TS:
                heat_abandon = H_CHP_GB - thermal_demand - H_TS
                heat_abandon = np.clip(heat_abandon, -self.max_tes_discharge, self.max_tes_charge)
                H_TS_abandon = self.tes_battery.charge(heat_abandon)
                H_TS += H_TS_abandon
                ope_step += self.tes_battery.operation_cost(H_TS_abandon)
                self.operation_cost += self.tes_battery.operation_cost(H_TS)
            else:
                heat_grid_sell = H_TS + thermal_demand - H_CHP_GB
                emission += self.grid_heat.carbon_emission(heat_grid_sell)
                self.carbon_emission += self.grid_heat.carbon_emission(heat_grid_sell)
                grid_step += self.grid_heat.sell(heat_grid_sell)
                self.grid_cost += self.grid_heat.sell(heat_grid_sell)
        else:  # discharging
            H_TS = -self.tes_battery.discharge(-H_TS)  # 放热为负数
            ope_step += self.tes_battery.operation_cost(-H_TS)
            self.operation_cost += self.tes_battery.operation_cost(-H_TS)

            if -H_TS + H_CHP_GB < thermal_demand:
                heat_grid_sell = thermal_demand + H_TS - H_CHP_GB
                emission += self.grid_heat.carbon_emission(heat_grid_sell)
                self.carbon_emission += self.grid_heat.carbon_emission(heat_grid_sell)
                grid_step += self.grid_heat.sell(heat_grid_sell)
                self.grid_cost += self.grid_heat.sell(heat_grid_sell)
            else:
                # 弃热量
                heat_abandon = -H_TS + H_CHP_GB - thermal_demand
                # 获取实际充热量
                heat_abandon = np.clip(abs(heat_abandon), -self.max_tes_discharge, self.max_tes_charge)
                H_TS_abandon = -self.tes_battery.discharge(heat_abandon)  # 放热为负数
                # 充热花费
                ope_step += self.tes_battery.operation_cost(H_TS_abandon)
                self.operation_cost += self.tes_battery.operation_cost(H_TS_abandon)
                H_TS += H_TS_abandon

        '''(5) 碳交易和绿证部分'''
        # 绿证交易收益，计算时要换算出mw
        self.gc_revenue += self.green_certificate.revenue(renewable / 1000, (P_CHP + renewable) / 1000)
        rev_step += self.green_certificate.revenue(renewable / 1000, (P_CHP + renewable) / 1000)

        # CHP和GB碳排放与碳配额
        emission += self.CHP.carbon_emission(H_CHP) + self.GB.carbon_emission(H_GB)
        quote = self.CHP.carbon_quote(H_CHP) + self.GB.carbon_quote(H_GB)
        self.carbon_emission += self.CHP.carbon_emission(H_CHP) + self.GB.carbon_emission(H_GB)

        # 碳排放成本
        carbon_step += self.pw_carbon.cost(emission, quote)
        self.carbon_cost += self.pw_carbon.cost(emission, quote)

        terminal = self.time_step == self.time_step_length - 1
        self.time_step = 0 if terminal else self.time_step + 1
        if terminal:
            self.reset()
        next_state = self._build_state()

        actual_cost = ope_step + fuel_step + carbon_step + rev_step + grid_step
        reward = -actual_cost
        info = [
            [elec_demand, P_CHP, renewable, P_ES, elec_grid_buy, elec_grid_sell],
            [thermal_demand, H_CHP, H_GB, H_TS, heat_grid_sell]
        ]

        # info = [[P_ES, P_CHP, solar_energy, EES_charge, EES_discharge,
        #         elec_grid_sell, elec_grid_buy, elec_demand, self.ees_battery.SoC,
        #         H_CHP, H_GB, TES_charge, TES_discharge, heat_grid_sell, heat_abandon,
        #         thermal_demand, self.tes_battery.SoC], emission, renewable, actual_cost]

        # self.print_interval += 1
        return next_state, reward, terminal, info

    def _reset_date(self) -> None:
        """
        Resets the date for the next episode.
        """
        if self.train == 1:
            self.day = random.choice(range(len(self.train_data_elec) // 24 - self.long_term_eval_days + 1))
            # self.day = random.randrange(len(self.train_data_elec) // 24 - self.long_term_eval_days + 1)
            # self.day = np.random.randint(len(self.train_data_elec) // 24 - self.long_term_eval_days + 1)
            elec_demand = self.train_data_elec[self.day * 24: (self.day + self.long_term_eval_days-1) * 24 + 24]
            heat_demand = self.train_data_heat[self.day * 24: (self.day + self.long_term_eval_days-1) * 24 + 24]
            solar_gen = self.train_data_solar[self.day * 24: (self.day + self.long_term_eval_days-1) * 24 + 24]
        elif self.train == 0:  # GP 使用
            if self.idx != 0:
                self.day = self.idx
            # self.day = np.random.randint(len(self.test_data_elec) // 24)
            elec_demand = self.train_data_elec[self.day * 24: (self.day + self.long_term_eval_days-1) * 24 + 24]
            heat_demand = self.train_data_heat[self.day * 24: (self.day + self.long_term_eval_days-1) * 24 + 24]
            solar_gen = self.train_data_solar[self.day * 24: (self.day + self.long_term_eval_days-1) * 24 + 24]
        elif self.train == -1:
            if self.idx != 0:
                self.day = self.idx
            elec_demand = self.test_data_elec[self.day * 24: (self.day + self.long_term_eval_days-1) * 24 + 24]
            heat_demand = self.test_data_heat[self.day * 24: (self.day + self.long_term_eval_days-1) * 24 + 24]
            solar_gen = self.test_data_solar[self.day * 24: (self.day + self.long_term_eval_days-1) * 24 + 24]
        self.users.electric_demand = elec_demand
        self.users.thermal_demand = heat_demand
        self.solar_generation.power = solar_gen

    def reset(self, seed=None, options=None) -> np.ndarray:
        self._reset_date()
        self.ees_battery.reset()
        self.tes_battery.reset()
        self.time_step = 0

        self.carbon_cost = 0.
        self.grid_cost = 0.
        self.generation_cost = 0.
        self.operation_cost = 0.
        self.fuel_cost = 0.
        self.gc_revenue = 0.
        self.ramp_cost = 0.
        self.balance_cost = 0.
        self.chp_prev = 0.
        self.gb_prev = 0.
        self.carbon_emission = 0.
        self.solar_use = 0.
        return self._build_state()

    def render(self, train=""):
        pass

    def close(self):
        """
        Nothing to be done here, but has to be defined
        """
        return

    def seed(self, s):
        """
        Set the random seed for consistent experiments
        """
        random.seed(s)
        np.random.seed(s)

class Fixed_Carbon:
    def __init__(self, pos_price, neg_price):
        self.carbon_price = pos_price
        self.carbon_profit = neg_price

    def carbon_trade_cost(self, emission, quote):
        pos = emission - quote
        if pos >= 0:
            return self.carbon_price * pos
        else:
            return self.carbon_profit * pos


class Ladder_Carbon:
    def __init__(self, interval, base_price, punishment, reward):
        self.interval = interval
        self.base_price = base_price
        self.punishment_coeff = punishment
        self.reward_coefficient = reward

    def carbon_trade_cost(self, emission, quote):
        pos = emission - quote
        if pos <= -self.interval:
            carbon_cost = -self.base_price * (1 + 2 * self.reward_coefficient) * (-pos - self.interval) - \
                          self.base_price * (1 + self.reward_coefficient) * self.interval
        elif -self.interval < pos <= 0:
            carbon_cost = -self.base_price * (1 + self.reward_coefficient)
        elif 0 < pos <= self.interval:
            carbon_cost = self.base_price * pos
        elif self.interval < pos <= 2 * self.interval:
            carbon_cost = self.base_price * self.interval + self.base_price * (1 + self.punishment_coeff) * (
                    pos - self.interval)
        else:
            carbon_cost = self.base_price * (2 + self.punishment_coeff) * self.interval + self.base_price * (
                    1 + 2 * self.punishment_coeff) * (pos - 2 * self.interval)
        return carbon_cost

    def set_interval(self, interval):
        self.interval = interval


class PW_Carbon:
    """
    根据碳配额制定分段碳交易价格，然后根据碳排放量计算碳成本
    """

    def __init__(self, interval, base_price, base_slop, reward_coeffi):
        self.interval = interval
        self.base_price = base_price
        self.b_slop = base_slop  # 作为碳成本的斜率
        self.reward = reward_coeffi  # 作为碳收益的斜率

    def carbon_price(self, pos):
        """
        根据碳配额计算碳价格
        :param pos: carbon price in specific position
        :return: carbon price
        """
        if pos < -self.interval:
            return -(1 + self.reward) * self.base_price + self.b_slop * (pos + self.interval)
        elif -self.interval <= pos < 0:
            return -(1 + self.reward) * self.base_price
        elif 0 <= pos < self.interval:
            return self.base_price
        elif self.interval <= pos < self.interval * 2:
            return self.base_price + self.b_slop * (pos - self.interval)
        else:  # 点斜式计算得到
            return self.b_slop * self.interval + self.base_price + 1.2 * self.b_slop * (pos - 2 * self.interval)

    def cost(self, emission, quote):
        import warnings
        from scipy import integrate
        pos = emission - quote
        if emission <= quote:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", integrate.IntegrationWarning)
                c, err = integrate.quad(self.carbon_price, pos, 0)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", integrate.IntegrationWarning)
                c, err = integrate.quad(self.carbon_price, 0, pos)
        return c


# 绿证模型
class green_certificate:
    def __init__(self, delta_g, quote_effi):
        self.delta = delta_g
        self.effi = quote_effi

    def revenue(self, power, total):
        '''
        :param power: 单位mw
        :param total: 单位mw
        :return: 正数表示光伏产量高于配额从而获得收益，负数表示购买额外绿证的成本
        '''
        return self.delta * (power - self.quote(total))  # return: $

    def quote(self, total):
        '''
        :param total: 总发电量，包括CHP，PV
        :return:
        '''
        return total * self.effi


class ES:
    def __init__(self, capacity, rateD, rateC, max_soc, min_soc, ope_effi):
        self.capacity = capacity  # full charge battery capacity
        self.rateD = rateD  # discharge rate
        self.rateC = rateC  # charging rate
        self.cur_capacity = 0.5 * self.capacity  # 当前capacity
        self.RC = self.capacity - self.cur_capacity  # 空余capacity  二者相加等于整个存储空间
        self.max_soc = max_soc
        self.min_soc = min_soc
        self.ope_effi = ope_effi

    def charge(self, E):
        if self.RC <= 0:
            self.RC = 0
            return 0
        else:
            if self.RC >= self.rateC * E:
                self.RC -= self.rateC * E  # 充电后剩余容量
                self.RC = max(self.RC, 0)
                self.cur_capacity += self.rateC * E  # 充电后当前容量
                self.cur_capacity = min(self.cur_capacity, self.capacity)
                self.update_soc()
                return self.rateC * E
            else:
                temp = self.RC
                self.RC = 0
                self.cur_capacity = self.capacity
                self.update_soc()
                return temp

    def discharge(self, E):
        if self.cur_capacity < self.min_soc * self.capacity:
            raise Exception("Wrong SoC range!")
        remaining = self.cur_capacity
        self.cur_capacity -= min(E, remaining - self.min_soc * self.capacity)
        self.RC += min(E, remaining - self.min_soc * self.capacity)

        self.cur_capacity = max(self.cur_capacity, self.min_soc * self.capacity)
        self.RC = min(self.RC, self.capacity)
        self.update_soc()
        return min(E, remaining - self.min_soc * self.capacity) * self.rateD  # 返回放电量 单位：KW

    def operation_cost(self, energy):  # 2.91$/KW
        return energy * self.ope_effi

    def update_soc(self):
        self.SoC = self.cur_capacity / self.capacity

    @property
    def SoC(self):
        soc = self.cur_capacity / self.capacity
        assert self.min_soc <= soc <= self.max_soc, "SoC wrong range: {}".format(soc)
        return soc

    def reset(self):
        self.cur_capacity = 0.5 * self.capacity
        self.RC = 0.5 * self.capacity

    @SoC.setter
    def SoC(self, value):
        self._SoC = value


# GB和CHP模型（不包括碳排放计算模型）
class GB:
    def __init__(self, alpha_gb, gas_price, ope_coeffi, quote_effi, CARBON_UNIT_GB):
        self.alpha = alpha_gb
        self.gas_price = gas_price
        self.ope_coeffi = ope_coeffi
        self.quote = quote_effi
        self.CARBON_UNIT_GB = CARBON_UNIT_GB

    def gas_consumed(self, heat):  # KW->m3
        return heat / self.alpha

    # 由于热能无法返网，因此可以考虑弃热成本
    def base_cost(self, heat):
        cost = self.gas_price * self.gas_consumed(heat)
        return cost  # $

    def operation_cost(self, heat):
        cost = heat * self.ope_coeffi
        return cost

    def carbon_emission(self, power_generated):
        return power_generated * self.CARBON_UNIT_GB

    def carbon_quote(self, heat):
        return heat * self.quote


# CHP的燃料为天燃气
class CHP:
    def __init__(self, alpha_chp, params, quote_effi, CARBON_UNIT_CHP):
        self.alpha = alpha_chp
        self.dg_a, self.dg_b, self.dg_c = params
        self.quote = quote_effi
        self.CARBON_UNIT_CHP = CARBON_UNIT_CHP

    def heat_generated(self, power_generated):
        return self.alpha * power_generated

    def base_ope_cost(self, power):  # 26$/KW  这里包含了电、热功率
        COST_CHP = self.dg_a + self.dg_b * power + self.dg_c * power ** 2
        return COST_CHP

    def carbon_emission(self, power_generated):
        return power_generated * self.CARBON_UNIT_CHP

    def carbon_quote(self, heat):
        return self.quote * heat


class Grid_Electric:
    def __init__(self, base_price, carbon_unit):
        self.base_price = base_price
        self.unit = carbon_unit
        self.time = 0

    def sell(self, E, time):
        """
        电网销售电能
        :param E:
        :return:
        """
        return self.base_price[time] * E

    def buy(self, E, time):
        """
        剩余电量上网
        :param E:
        :return:
        """
        return -self.base_price[time] * 0.8 * E  # 折扣系数为0.8

    # 使用主网的电能需要计及额外碳排放。
    def carbon_emission(self, E):
        return E * self.unit


class Grid_Heat:
    def __init__(self, price, carbon_unit):
        self.sell_prices = price
        self.unit = carbon_unit  # t/KW

    def sell(self, H):
        return self.sell_prices * H  # 用户从气网购热需要支付额外费用

    def carbon_emission(self, H):  # 从热网购热需要额外碳成本
        return self.unit * H


# solar and wind from https://doi.org/10.1109/JIOT.2020.2966232
class Generation_Solar:
    """
    For wind and solar power
    """

    def __init__(self, generation, ope_coeff):
        self.power = generation
        self.max_power = np.max(generation)
        self.min_power = np.min(generation)
        self.ope_coeff = ope_coeff

    def current_generation(self, time):
        return self.power[time]

    def operation_cost(self, power):
        return self.ope_coeff * power


class User:
    def __init__(self, electric_demand, thermal_demand):
        # 第一次初始化时为全部数据，其余更新为当天数据
        self.electric_demand = electric_demand
        self.thermal_demand = thermal_demand
        # 初始化时记录最大、最小值
        self.max_electric_demand = np.max(electric_demand)
        self.min_electric_demand = np.min(electric_demand)
        self.max_thermal_demand = np.max(thermal_demand)
        self.min_thermal_demand = np.min(thermal_demand)

    def current_electric_demand(self, time):
        demand = self.electric_demand[time]
        return demand

    def current_thermal_demand(self, time):
        return self.thermal_demand[time]


class PowerNetEnv(gym.Env):
    """
        Custom Environment for Power Network Management.

        The environment simulates a power network, and the agent's task is to
        manage this network by controlling the batteries attached to various nodes.

        Attributes:
            voltage_limits (tuple): Limits for the voltage.
            algorithm (str): Algorithm choice. Can be 'Laurent' or 'PandaPower'.
            battery_list (list): List of nodes where batteries are attached.
            year (int): Current year in simulation.
            month (int): Current month in simulation.
            day (int): Current day in simulation.
            train (bool): Whether the environment is in training mode.
            state_pattern (str): Pattern for the state representation.
            network_info (dict): Information about the network.
            node_num (int): Number of nodes in the network.
            action_space (gym.spaces.Box): Action space of the environment.
            data_manager (GeneralPowerDataManager): Manager for the time-series data.
            episode_length (int): Length of an episode.
            state_length (int): Length of the state representation.
            state_min (np.ndarray): Minimum values for each state element.
            state_max (np.ndarray): Maximum values for each state element.
            state_space  (gym.spaces.Box): State space of the environment.
            current_time (int): Current timestep in the episode.
            after_control (np.ndarray): Voltages after control is applied.

        Args:
            env_config_path (str): Path to the environment configuration file.

        """

    def __init__(self, env_config: dict = env_config) -> None:
        """
         Initialize the PowerNetEnv-before environment.
         :param env_config_path: Path to the environment configuration file. Defaults to 'env_config.py'.
         :type env_config_path: str
         """
        config = env_config
        self.is_split = env_config['is_split']

        self.voltage_low_boundary = config['voltage_limits'][0]
        self.voltage_high_boundary = config['voltage_limits'][1]
        self.algorithm = config['algorithm']
        self.battery_list = config['battery_list']
        self.seed = config['seed']
        self.year = config['year']
        self.month = config['month']
        self.day = config['day']
        self.train = config['train']
        self.rand = config['seed']
        self.state_pattern = config['state_pattern']
        self.network_info = config['network_info']
        self.long_term_eval_days = config['long_term_eval_days']
        self.Discrete_value = config['Discrete_value']

        # network_info for building the network
        if self.network_info == 'None':
            print('create basic 34 node IEEE network, when initial data is not identified')
            self.network_info = {'vm_pu': 1.0, 's_base': 1000,
                                 'bus_info_file': '../data_sources/network_data/node_34/Nodes_34.csv',
                                 'branch_info_file': '../data_sources/network_data/node_34/Lines_34.csv'}
            self.s_base = 1000
            self.node_num = 34
        else:
            self.s_base = self.network_info['s_base']
            network_bus_info = pd.read_csv(self.network_info['bus_info_file'])
            self.node_num = len((network_bus_info.NODES))
        # Conditional initialization of the distribution network based on the chosen algorithm
        if self.algorithm == "Laurent":
            # Logic for initializing with GridTensor
            self.net = GridTensor(self.network_info['bus_info_file'], self.network_info['branch_info_file'])
            self.net.Q_file = np.zeros(self.node_num-1)
            self.dense_Ybus = self.net._make_y_bus().toarray()
        elif self.algorithm == "PandaPower":
            # Logic for initializing with PandaPower
            self.net = create_pandapower_net(self.network_info)
        else:
            raise ValueError("Invalid algorithm choice. Please choose 'Laurent' or 'PandaPower'.")

        if not self.battery_list:
            raise ValueError("No batteries specified!")

        battery_parameters['is_split'] = self.is_split
        for node_index in self.battery_list:
            battery = Battery(battery_parameters)
            setattr(self, f"battery_{node_index}", battery)
        self.batteries = [getattr(self, f"battery_{i}") for i in self.battery_list]
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.battery_list), ), dtype=np.float32)
        self.data_manager = GeneralPowerDataManager(config['time_series_data_path'], self.long_term_eval_days)
        self.episode_length: int = self.long_term_eval_days * 24 * 60 / self.data_manager.time_interval

        # self.train_day_num = len(self.data_manager.train_dates)
        # self.test_day_num = len(self.data_manager.test_dates)

        self.train_day_num = len(self.data_manager.limit_train_dates)
        self.test_day_num = len(self.data_manager.limit_test_dates)

        self.idx = 0  # 为了GP设置的变量

        if self.state_pattern == 'default':
            self.state_length = len(self.battery_list) * 3 + self.node_num + 2
            print(self.data_manager.active_power_min)
            print(self.data_manager.price_min)
            self.state_min = np.array([self.data_manager.active_power_min, 0.2, self.data_manager.price_min, 0.0, 0.5, self.data_manager.renewable_active_power_min])
            self.state_max = np.array(
                [self.data_manager.active_power_max, 0.8, self.data_manager.price_max, self.episode_length - 1, 1.5, self.data_manager.renewable_active_power_max])
        else:
            raise ValueError("Invalid value for 'state_pattern'. Expected 'default' or define by yourself.")

        self.state_space = spaces.Box(low=-2, high=2, shape=(self.state_length,), dtype=np.float32)

    def reset(self, seed=None, options=None) -> np.ndarray:
        """
        Reset the environment to its initial state and return the initial state.

        :return: The normalized initial state of the environment.
        :rtype: np.ndarray
        """
        # super().reset()
        self._reset_date()
        self._reset_time()
        self._reset_batteries()

        return self._build_state()

    def _reset_date(self) -> None:
        """
        Resets the date for the next episode.
        """
        if self.train == 1:
            # self.year, self.month, self.day = random.choice(self.data_manager.train_dates)
            self.year, self.month, self.day = random.choice(self.data_manager.limit_train_dates)
        elif self.train == 0:
            # self.year, self.month, self.day = random.choice(self.data_manager.test_dates)
            # 为了GP保留最优个体，self.train = False, 从self.data_manager.train_dates 按顺序测试个体
            # self.year, self.month, self.day = self.data_manager.train_dates[self.idx]
            self.year, self.month, self.day = self.data_manager.limit_train_dates[self.idx]
        elif self.train == -1:
            # self.year, self.month, self.day = self.data_manager.test_dates[self.idx]
            self.year, self.month, self.day = self.data_manager.limit_test_dates[self.idx]
    def _reset_time(self) -> None:
        """
        Resets the time for the next episode.
        """
        self.current_time = 0

    def _reset_batteries(self) -> None:
        """
        Resets the batteries for the next episode.
        """
        # for node_index in self.battery_list:
        #     getattr(self, f"battery_{node_index}").reset()
        for i in range(len(self.battery_list)):
            self.batteries[i].reset()

    def _build_state(self) -> np.ndarray:
        """
        Builds the current state of the environment based on the current time and data from PowerDataManager.

        Returns:
            normalized_state (np.ndarray): The current state of the environment, normalized between 0 and 1.
                The state includes the following variables:
                - Netload power
                - SOC (State of Charge) of the last battery in the battery list
                - Price of the energy
                - Time state of the day
                - Voltage from estimation
        """
        obs = self._get_obs()
        if self.state_pattern == 'default':
            active_power = np.array(list(obs['node_data']['active_power'].values()))
            price = obs['price']
            soc_list = np.array([
                float(obs['battery_data']['soc'][f'battery_{node_index}'][0]) if isinstance(
                    obs['battery_data']['soc'][f'battery_{node_index}'], np.ndarray)
                else float(obs['battery_data']['soc'][f'battery_{node_index}'])
                for node_index in self.battery_list
            ])

            vm_pu_battery = np.array(
                [obs['node_data']['voltage'][f'node_{node_index}'] for node_index in self.battery_list])
            battery_renewable_power = np.array(
                [obs['node_data']['renewable_active_power'][f'node_{node_index}'] for node_index in self.battery_list]
            )
            state = np.concatenate((active_power, soc_list, [price], [self.current_time], vm_pu_battery, battery_renewable_power))
            self.state = state
            normalized_state = self._normalize_state(state)
            self.normalized_state = normalized_state
        return normalized_state

    def _split_state(self, state):
        net_load_length = self.node_num
        num_batteries = len(self.battery_list)

        soc_all_length = num_batteries
        vm_pu_battery_nodes_length = num_batteries

        soc_all_start = net_load_length
        price_start = soc_all_start + soc_all_length
        current_time_start = price_start + 1
        vm_pu_battery_nodes_start = current_time_start + 1

        net_load = state[:net_load_length]
        soc_all = state[soc_all_start:soc_all_start + soc_all_length]
        price = np.array([state[price_start]])
        current_time = np.array([state[current_time_start]])
        vm_pu_battery_nodes = state[vm_pu_battery_nodes_start:]

        return net_load, soc_all, price, current_time, vm_pu_battery_nodes

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalizes the state variables.

        Parameters:
            state (np.ndarray): The current state of the environment.

        Returns:
            np.ndarray: The normalized state of the environment.
        """
        # action_power
        state[:self.node_num] = (state[:self.node_num] - self.state_min[0]) / (self.state_max[0] - self.state_min[0])
        # soc_list
        state[self.node_num:self.node_num + len(self.battery_list)] = \
            (state[self.node_num:self.node_num + len(self.battery_list)] - self.state_min[1]) / (self.state_max[1] - self.state_min[1])
        # price
        state[self.node_num + len(self.battery_list):self.node_num + len(self.battery_list) + 1] = \
            ((state[self.node_num + len(self.battery_list):self.node_num + len(self.battery_list) + 1] - self.state_min[2]) / (self.state_max[2] - self.state_min[2]))
        # current_time
        state[self.node_num + len(self.battery_list) + 1:self.node_num + len(self.battery_list) + 2] = \
            ((state[self.node_num + len(self.battery_list) + 1:self.node_num + len(self.battery_list) + 2] -self.state_min[3]) / (self.state_max[3] -self.state_min[3]))
        # battery_renewable
        state[self.node_num + 2*len(self.battery_list) + 2: self.node_num + 3* len(self.battery_list) + 2] = \
            ((state[self.node_num + 2*len(self.battery_list) + 2: self.node_num + 3* len(self.battery_list) + 2] -
              self.state_min[5]) / (self.state_max[5] - self.state_min[5]))
        normalized_state = state
        return normalized_state

    def _denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """
        Denormalizes the state variables.

        Parameters:
            normalized_state (np.ndarray): The normalized state of the environment.

        Returns:
            np.ndarray: The denormalized state of the environment.
        """
        normalized_state[:self.node_num] = normalized_state[:self.node_num] * (self.state_max[0] - self.state_min[0]) + self.state_min[0]

        normalized_state[self.node_num:self.node_num + len(self.battery_list)] = \
            (normalized_state[self.node_num:self.node_num + len(self.battery_list)] * (self.state_max[1] -self.state_min[1]) + self.state_min[1])
        normalized_state[self.node_num + len(self.battery_list):self.node_num + len(self.battery_list) + 1] = \
            (normalized_state[self.node_num + len(self.battery_list):self.node_num + len(self.battery_list) + 1] * (self.state_max[2] -self.state_min[2]) + self.state_min[2])
        normalized_state[self.node_num + len(self.battery_list) + 1:self.node_num + len(self.battery_list) + 2] = (
                normalized_state[self.node_num + len(self.battery_list) + 1:self.node_num + len(self.battery_list) + 2] * (self.state_max[3] -self.state_min[3]) + self.state_min[3])
        denormalized_state = normalized_state
        return denormalized_state

    def _get_obs(self):
        """
        Executes the power flow based on the chosen algorithm and returns the observations.

        Returns:
            dict: The observation dictionary containing various state elements.
        """
        node_num = self.node_num
        if self.state_pattern == 'default':
            one_slot_data = self.data_manager.select_timeslot_data(self.year, self.month, self.day, self.current_time)

            if self.algorithm == "Laurent":
                # This is where bugs comes from, if we don't use copy, this slice is actually creating a view of originally data.
                active_power = cp.copy(one_slot_data[0:node_num])
                renewable_active_power = one_slot_data[node_num:node_num*2]
                self.active_power = (active_power - renewable_active_power)[1:node_num]
                reactive_power = np.zeros(33)
                price = one_slot_data[-1]
                self.solution = self.net.run_pf(active_power=self.active_power)

                obs = {'node_data': {'voltage': {}, 'active_power': {}, 'reactive_power': {}, 'renewable_active_power': {}},
                       'battery_data': {'soc': {}},
                       'price': {},
                       'aux': {}
                }

                for node_index in range(len(self.net.bus_info.NODES)):  # NODES[1-34], node_index[0-33]
                    if node_index == 0:
                        obs['node_data']['voltage'][f'node_{node_index}'] = 1.0
                        obs['node_data']['active_power'][f'node_{node_index}'] = 0.0
                        obs['node_data']['renewable_active_power'][f'node_{node_index}'] = 0.0
                    else:
                        obs['node_data']['voltage'][f'node_{node_index}'] = abs(
                            self.solution['v'].T[node_index - 1]).squeeze()
                        obs['node_data']['active_power'][f'node_{node_index}'] = active_power[node_index - 1]
                        obs['node_data']['renewable_active_power'][f'node_{node_index}'] = renewable_active_power[
                            node_index - 1]
                for node_index in self.battery_list:
                    obs['battery_data']['soc'][f'battery_{node_index}'] = getattr(self, f'battery_{node_index}').SOC()
                obs['price'] = price
            else:
                active_power = one_slot_data[0:34]
                active_power[0] = 0
                renewable_active_power = one_slot_data[34:68]
                renewable_active_power[0] = 0
                price = one_slot_data[-1]
                for bus_index in self.net.load.bus.index:
                    self.net.load.p_mw[bus_index] = (active_power[bus_index] - renewable_active_power[
                        bus_index]) / self.s_base
                    self.net.load.q_mvar[bus_index] = 0
                pp.runpp(self.net, algorithm='nr')
                v_real = self.net.res_bus["vm_pu"].values * np.cos(np.deg2rad(self.net.res_bus["va_degree"].values))
                v_img = self.net.res_bus["vm_pu"].values * np.sin(np.deg2rad(self.net.res_bus["va_degree"].values))
                v_result = v_real + 1j * v_img

                obs = {'node_data': {'voltage': {}, 'active_power': {}, 'reactive_power': {},
                                     'renewable_active_power': {}}, 'battery_data': {'soc': {}}, 'price': {}, 'aux': {}}

                for node_index in self.net.load.bus.index:
                    bus_idx = self.net.load.at[node_index, 'bus']
                    obs['node_data']['voltage'][f'node_{node_index}'] = self.net.res_bus.vm_pu.at[bus_idx]
                    obs['node_data']['active_power'][f'node_{node_index}'] = active_power[node_index]
                    obs['node_data']['reactive_power'][f'node_{node_index}'] = self.net.res_load.q_mvar[node_index]
                    obs['node_data']['renewable_active_power'][f'node_{node_index}'] = renewable_active_power[
                        node_index]
                for node_index in self.battery_list:
                    obs['battery_data']['soc'][f'battery_{node_index}'] = getattr(self, f'battery_{node_index}').SOC()
                obs['price'] = price
        else:
            raise ValueError('please redesign the get obs function to fit the pattern you want')
        return obs

    def _apply_battery_actions(self, action):
        '''apply action to battery charge/discharge, update the battery condition, excute power flow, update the network condition'''
        if self.state_pattern == 'default':
            if self.algorithm == "Laurent":
                v = self.solution["v"]
                v_totall = np.insert(v, 0, 1)
                current_each_node = np.matmul(self.dense_Ybus, v_totall)
                power_imported_from_ex_grid_before = current_each_node[0].real

                for i, battery in enumerate(self.batteries):
                    battery.step(action[i])
                    self.active_power[self.battery_list[i]-1] += battery.energy_change
                # for i, node_index in enumerate(self.battery_list):
                #     getattr(self, f"battery_{node_index}").step(action[i])
                #     self.active_power[node_index - 1] += getattr(self, f"battery_{node_index}").energy_change
                self.solution = self.net.run_pf(active_power=self.active_power)

                v = self.solution["v"]
                v_totall = np.insert(v, 0, 1)
                vm_pu_after_control = cp.deepcopy(abs(v_totall))
                vm_pu_after_control_bat = np.squeeze(vm_pu_after_control)[self.battery_list]
                self.after_control = vm_pu_after_control
                current_each_node = np.matmul(self.dense_Ybus, v_totall)
                power_imported_from_ex_grid_after = current_each_node[0].real
                saved_energy = power_imported_from_ex_grid_before - power_imported_from_ex_grid_after
            else:
                power_imported_from_ex_grid_before = cp.deepcopy(self.net.res_ext_grid['p_mw'])

                for i, node_index in enumerate(self.battery_list):
                    getattr(self, f"battery_{node_index}").step(action[i])
                    self.net.load.p_mw[node_index] += getattr(self, f"battery_{node_index}").energy_change / 1000
                pp.runpp(self.net, algorithm='nr')
                vm_pu_after_control = cp.deepcopy(self.net.res_bus.vm_pu).to_numpy(dtype=float)
                vm_pu_after_control_bat = vm_pu_after_control[self.battery_list]

                self.after_control = vm_pu_after_control
                power_imported_from_ex_grid_after = self.net.res_ext_grid['p_mw']
                saved_energy = power_imported_from_ex_grid_before - power_imported_from_ex_grid_after
        else:
            raise ValueError('Expected default or define yourself based on the goal')
        return saved_energy, vm_pu_after_control_bat

    def step(self, action: np.ndarray) -> tuple:
        """
        Advance the environment by one timestep based on the provided action.

        :param action: Action to execute.
        :type action: np.ndarray
        :return: Tuple containing the next normalized observation, the reward, a boolean indicating if the episode has ended, and additional info.
        :rtype: tuple
        """
        action = action.reshape(-1, 1)
        # print(f"Action received: {action}")

        current_normalized_obs = self.normalized_state
        # info = current_normalized_obs
        # Apply battery actions and get updated observations
        saved_energy, vm_pu_after_control_bat = self._apply_battery_actions(action)

        reward = self._calculate_reward(current_normalized_obs, vm_pu_after_control_bat, saved_energy)

        finish = (self.current_time == self.episode_length - 1)
        self.current_time += 1
        truncated = False
        if finish:
            self.current_time = 0
            next_normalized_obs = self.reset()
            terminated = True
        else:
            next_normalized_obs = self._build_state()
            terminated = False
        # info
        info = {"TimeLimit.truncated": truncated, "episode": self.current_time}
        return next_normalized_obs, float(reward), terminated, info

    def _calculate_reward(self, current_normalized_obs: np.ndarray, vm_pu_after_control_bat: np.ndarray,
                          saved_power: float) -> float:
        """
        Calculate the reward based on the current observation and saved power. the default version is to calculate the battey saved energy
        based on the current price

        Parameters:
            current_normalized_obs (np.ndarray): The current normalized observations.
            vm_pu_after_control_bat (np.ndarray): The voltage after control at battery locations.
            saved_power (float): The amount of power saved.

        Returns:
            float: Calculated reward.
        """
        if self.state_pattern == 'default':
            # 归一化后的reward，
            # current_normalized_obs[self.node_num + len(self.battery_list)] 电价被归一化了
            reward_for_power = 1 * current_normalized_obs[self.node_num + len(self.battery_list)] * float(saved_power)
            reward_for_penalty = 0.0
            # 如果节点电压超过[0.95,1.05], 那么惩罚值大于1
            for vm_pu_bat in vm_pu_after_control_bat:
                reward_for_penalty += min(0, 100 * (0.05 - abs(1.0 - vm_pu_bat)))

            self.reward_for_power = reward_for_power
            if reward_for_penalty > 0:
                print('penalty reward is {}'.format(reward_for_penalty))
            self.reward_for_penalty = reward_for_penalty

            # 反归一化后的reward
            self.saved_money = -1 * self._denormalize_state(current_normalized_obs)[self.node_num + len(self.battery_list)] * float(saved_power)

            reward = reward_for_power + reward_for_penalty
        else:
            raise ValueError(
                "Invalid value for 'state_pattern'. Expected 'default, or define by yourself based on different goal")

        return reward

    def render(self, current_obs, next_obs, reward, finish):
        """
        Render the environment's current state.

        :param current_obs: Current observation.
        :type current_obs: np.array
        :param next_obs: Next observation.
        :type next_obs: np.array
        :param reward: Reward obtained from the last action.
        :type reward: float
        :param finish: Whether the episode has ended.
        :type finish: bool
        """
        print('state={}, next_state={}, reward={:.4f}, terminal={}\n'.format(current_obs, next_obs, reward, finish))


if __name__ == '__main__':
    power_net_env = PowerNetEnv(env_config=env_config)
    power_net_env.reset()

    for j in range(1):
        episode_reward = 0
        for i in range(1000):
            # 1 is charge -1 is discharge
            # tem_action = np.ones(len(power_net_env.battery_list))
            tem_action = power_net_env.action_space.sample()
            tem_action = tem_action.ravel()
            # print(f'year, month, day, current time',
            #       (power_net_env.year, power_net_env.month, power_net_env.day, power_net_env.current_time))
            print(f'year, month, day, current time {(power_net_env.year, power_net_env.month, power_net_env.day, power_net_env.current_time)}')
            # print(f'current month is {power_net_env.month}, current day is {power_net_env.day}, current time is {power_net_env.current_time}')
            next_obs, reward, finish, info = power_net_env.step(tem_action)
            # print(power_net_env.reward_for_power)
            print(power_net_env.reward_for_penalty)
            # print('reward',reward)
            episode_reward += reward  # power_net_env.render(current_obs, next_obs, reward, finish)
        print(episode_reward)
