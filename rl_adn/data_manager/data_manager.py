"""
This module contains the DataManager class for managing and preprocessing
time-series data related to power systems. It includes functionalities for
data loading, cleaning, and basic manipulations.
"""
from typing import List, Tuple, Union
import pandas as pd
import numpy as np
import random
import re
import os


def MEMG_train_and_test_data(file_path, long_term_eval_days):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_path)
    file_path = os.path.normpath(file_path)

    # 读取数据
    Date = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=[0], dtype=str)  # 日期
    ELEC = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=[2])  # 用电量 (KW)
    SOLAR = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=[3])  # KW
    HEAT = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=[4])  # 供热量 (KW)

    month_days = [31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30]
    start_idx = 0  # 用于追踪当前数据的索引位置

    train_elec, test_elec = [], []
    train_heat, test_heat = [], []
    train_solar, test_solar = [], []
    train_date, test_date = [], []

    for days in month_days:
        train_days = 20
        train_elec.append(ELEC[start_idx: start_idx + train_days * 24])
        test_elec.append(ELEC[start_idx + train_days * 24: start_idx + days * 24])

        train_heat.append(HEAT[start_idx: start_idx + train_days * 24])
        test_heat.append(HEAT[start_idx + train_days * 24: start_idx + days * 24])

        train_solar.append(SOLAR[start_idx: start_idx + train_days * 24])
        test_solar.append(SOLAR[start_idx + train_days * 24: start_idx + days * 24])

        train_date.append(Date[start_idx: start_idx + train_days * 24])
        test_date.append(Date[start_idx + train_days * 24: start_idx + days * 24])

        start_idx += days * 24  # 更新索引

    # 合并所有月份的数据
    train_data_elec = np.concatenate(train_elec)
    test_data_elec = np.concatenate(test_elec)
    train_data_heat = np.concatenate(train_heat)
    test_data_heat = np.concatenate(test_heat)
    train_data_solar = np.concatenate(train_solar)
    test_data_solar = np.concatenate(test_solar)
    train_data_date = np.concatenate(train_date)
    test_data_date = np.concatenate(test_date)

    return (train_data_elec, train_data_heat, train_data_solar, train_data_date,
            test_data_elec, test_data_heat, test_data_solar, test_data_date)


def MEMG_train_and_test_data_after(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_path)
    file_path = os.path.normpath(file_path)
    Date = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=[0], dtype=str)  # Dates as strings
    ELEC = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=[2])  # KW
    HEAT = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=[4])  # KW
    SOLAR = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=[3])  # KW

    # 每个月前20天的数据用来训练，后10天的数据用来测试

    # 7 8 9 10 11 12 1 2 3 4 5 6
    train_1_elec = ELEC[0:20 * 24]  # 20天的数据
    train_2_elec = ELEC[31 * 24:51 * 24]
    train_3_elec = ELEC[62 * 24:82 * 24]
    train_4_elec = ELEC[92 * 24:112 * 24]
    train_5_elec = ELEC[123 * 24:143 * 24]
    train_6_elec = ELEC[153 * 24:173 * 24]
    train_7_elec = ELEC[184 * 24:204 * 24]
    train_8_elec = ELEC[215 * 24:235 * 24]
    train_9_elec = ELEC[243 * 24:263 * 24]
    train_10_elec = ELEC[274 * 24:294 * 24]
    train_11_elec = ELEC[304 * 24:324 * 24]
    train_12_elec = ELEC[335 * 24:355 * 24]

    test_1_elec = ELEC[20 * 24:31 * 24]  # 10天
    test_2_elec = ELEC[51 * 24:62 * 24]
    test_3_elec = ELEC[82 * 24:92 * 24]
    test_4_elec = ELEC[112 * 24:123 * 24]
    test_5_elec = ELEC[143 * 24:153 * 24]
    test_6_elec = ELEC[173 * 24:184 * 24]
    test_7_elec = ELEC[204 * 24:215 * 24]
    test_8_elec = ELEC[235 * 24:243 * 24]
    test_9_elec = ELEC[263 * 24:274 * 24]
    test_10_elec = ELEC[294 * 24:304 * 24]
    test_11_elec = ELEC[324 * 24:335 * 24]
    test_12_elec = ELEC[355 * 24:365 * 24]

    train_data_elec = np.concatenate(
        [train_1_elec, train_2_elec, train_3_elec, train_4_elec, train_5_elec, train_6_elec, train_7_elec,
         train_8_elec, train_9_elec, train_10_elec, train_11_elec, train_12_elec])

    test_data_elec = np.concatenate(
        [test_1_elec, test_2_elec, test_3_elec, test_4_elec, test_5_elec, test_6_elec, test_7_elec,
         test_8_elec, test_9_elec, test_10_elec, test_11_elec, test_12_elec])

    train_1_heat = HEAT[0: 20 * 24]
    train_2_heat = HEAT[31 * 24:51 * 24]
    train_3_heat = HEAT[62 * 24:82 * 24]
    train_4_heat = HEAT[92 * 24:112 * 24]
    train_5_heat = HEAT[123 * 24:143 * 24]
    train_6_heat = HEAT[153 * 24:173 * 24]
    train_7_heat = HEAT[184 * 24:204 * 24]
    train_8_heat = HEAT[215 * 24:235 * 24]
    train_9_heat = HEAT[243 * 24:263 * 24]
    train_10_heat = HEAT[274 * 24:294 * 24]
    train_11_heat = HEAT[304 * 24:324 * 24]
    train_12_heat = HEAT[335 * 24:355 * 24]

    test_1_heat = HEAT[20 * 24:31 * 24]
    test_2_heat = HEAT[51 * 24:62 * 24]
    test_3_heat = HEAT[82 * 24:92 * 24]
    test_4_heat = HEAT[112 * 24:123 * 24]
    test_5_heat = HEAT[143 * 24:153 * 24]
    test_6_heat = HEAT[173 * 24:184 * 24]
    test_7_heat = HEAT[204 * 24:215 * 24]
    test_8_heat = HEAT[235 * 24:243 * 24]
    test_9_heat = HEAT[263 * 24:274 * 24]
    test_10_heat = HEAT[294 * 24:304 * 24]
    test_11_heat = HEAT[324 * 24:335 * 24]
    test_12_heat = HEAT[355 * 24:365 * 24]

    train_data_heat = np.concatenate(
        [train_1_heat, train_2_heat, train_3_heat, train_4_heat, train_5_heat, train_6_heat, train_7_heat,
         train_8_heat, train_9_heat, train_10_heat, train_11_heat, train_12_heat])

    test_data_heat = np.concatenate(
        [test_1_heat, test_2_heat, test_3_heat, test_4_heat, test_5_heat, test_6_heat, test_7_heat,
         test_8_heat, test_9_heat, test_10_heat, test_11_heat, test_12_heat])

    train_1_date = Date[0:20 * 24]  # 20天的数据
    train_2_date = Date[31 * 24:51 * 24]
    train_3_date = Date[62 * 24:82 * 24]
    train_4_date = Date[92 * 24:112 * 24]
    train_5_date = Date[123 * 24:143 * 24]
    train_6_date = Date[153 * 24:173 * 24]
    train_7_date = Date[184 * 24:204 * 24]
    train_8_date = Date[215 * 24:235 * 24]
    train_9_date = Date[243 * 24:263 * 24]
    train_10_date = Date[274 * 24:294 * 24]
    train_11_date = Date[304 * 24:324 * 24]
    train_12_date = Date[335 * 24:355 * 24]

    test_1_date = Date[20 * 24:31 * 24]  # 10天
    test_2_date = Date[51 * 24:62 * 24]
    test_3_date = Date[82 * 24:92 * 24]
    test_4_date = Date[112 * 24:123 * 24]
    test_5_date = Date[143 * 24:153 * 24]
    test_6_date = Date[173 * 24:184 * 24]
    test_7_date = Date[204 * 24:215 * 24]
    test_8_date = Date[235 * 24:243 * 24]
    test_9_date = Date[263 * 24:274 * 24]
    test_10_date = Date[294 * 24:304 * 24]
    test_11_date = Date[324 * 24:335 * 24]
    test_12_date = Date[355 * 24:365 * 24]

    train_data_date = np.concatenate(
        [train_1_date, train_2_date, train_3_date, train_4_date, train_5_date, train_6_date, train_7_date,
         train_8_date, train_9_date, train_10_date, train_11_date, train_12_date])

    test_data_date = np.concatenate(
        [test_1_date, test_2_date, test_3_date, test_4_date, test_5_date, test_6_date, test_7_date,
         test_8_date, test_9_date, test_10_date, test_11_date, test_12_date])

    train_1_solar = SOLAR[0: 20 * 24]
    train_2_solar = SOLAR[31 * 24:51 * 24]
    train_3_solar = SOLAR[62 * 24:82 * 24]
    train_4_solar = SOLAR[92 * 24:112 * 24]
    train_5_solar = SOLAR[123 * 24:143 * 24]
    train_6_solar = SOLAR[153 * 24:173 * 24]
    train_7_solar = SOLAR[184 * 24:204 * 24]
    train_8_solar = SOLAR[215 * 24:235 * 24]
    train_9_solar = SOLAR[243 * 24:263 * 24]
    train_10_solar = SOLAR[274 * 24:294 * 24]
    train_11_solar = SOLAR[304 * 24:324 * 24]
    train_12_solar = SOLAR[335 * 24:355 * 24]

    test_1_solar = SOLAR[20 * 24:31 * 24]
    test_2_solar = SOLAR[51 * 24:62 * 24]
    test_3_solar = SOLAR[82 * 24:92 * 24]
    test_4_solar = SOLAR[112 * 24:123 * 24]
    test_5_solar = SOLAR[143 * 24:153 * 24]
    test_6_solar = SOLAR[173 * 24:184 * 24]
    test_7_solar = SOLAR[204 * 24:215 * 24]
    test_8_solar = SOLAR[235 * 24:243 * 24]
    test_9_solar = SOLAR[263 * 24:274 * 24]
    test_10_solar = SOLAR[294 * 24:304 * 24]
    test_11_solar = SOLAR[324 * 24:335 * 24]
    test_12_solar = SOLAR[355 * 24:365 * 24]

    train_data_solar = np.concatenate(
        [train_1_solar, train_2_solar, train_3_solar, train_4_solar, train_5_solar, train_6_solar, train_7_solar,
         train_8_solar, train_9_solar, train_10_solar, train_11_solar, train_12_solar])

    test_data_solar = np.concatenate(
        [test_1_solar, test_2_solar, test_3_solar, test_4_solar, test_5_solar, test_6_solar, test_7_solar,
         test_8_solar, test_9_solar, test_10_solar, test_11_solar, test_12_solar])
    return train_data_elec, train_data_heat, train_data_solar, train_data_date, test_data_elec, test_data_heat, test_data_solar, test_data_date


class GeneralPowerDataManager:
    """
    A class to manage and preprocess time series data for power systems.

    Attributes:
        df (pd.DataFrame): The original data.
        data_array (np.ndarray): Array representation of the data.
        active_power_cols (List[str]): List of columns related to active power.
        reactive_power_cols (List[str]): List of columns related to reactive power.
        renewable_active_power_cols (List[str]): List of columns related to renewable active power.
        renewable_reactive_power_cols (List[str]): List of columns related to renewable reactive power.
        price_col (List[str]): List of columns related to price.
        train_dates (List[Tuple[int, int, int]]): List of training dates.
        test_dates (List[Tuple[int, int, int]]): List of testing dates.
        time_interval (int): Time interval of the data in minutes.
    """

    def __init__(self, datapath: str, long_term_eval_days: int) -> None:
        """
        Initialize the GeneralPowerDataManager object.

        Parameters:
            datapath (str): Path to the CSV file containing the data.
        """
        self.long_term_eval_days = long_term_eval_days
        if datapath is None:
            raise ValueError("Please input the correct datapath")

        data = pd.read_csv(datapath)

        # Check if 'date_time' column exists
        if 'date_time' in data.columns:
            data.set_index('date_time', inplace=True)
        else:
            first_col = data.columns[0]
            data.set_index(first_col, inplace=True)

        data.index = pd.to_datetime(data.index)

        # Print data scale and initialize time interval
        min_date = data.index.min()
        max_date = data.index.max()
        print(f"Data scale: from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

        self.time_interval = int((data.index[1] - data.index[0]).seconds / 60)
        print(f"Data time interval: {self.time_interval} minutes")

        # Initialize other attributes
        self.df = data
        self.data_array = data.values

        self.active_power_cols = [col for col in self.df.columns if re.fullmatch(r'active_power(_\w+)?', col)]
        self.reactive_power_cols = [col for col in self.df.columns if re.fullmatch(r'reactive_power(_\w+)?', col)]
        self.renewable_active_power_cols = [col for col in self.df.columns if re.fullmatch(r'renewable_active_power(_\w+)?', col)]
        self.renewable_reactive_power_cols = [col for col in self.df.columns if re.fullmatch(r'renewable_reactive_power(_\w+)?', col)]
        self.price_col = [col for col in self.df.columns if re.fullmatch(r'price(_\w+)?', col)]
        # Display dataset information
        print(f"Dataset loaded from {datapath}")
        print(f"Dataset dimensions: {self.df.shape}")
        print(f"Dataset contains the following types of data:")
        print(
            f"Active power columns: {self.active_power_cols} (Indices: {[self.df.columns.get_loc(col) for col in self.active_power_cols]})")
        print(
            f"Reactive power columns: {self.reactive_power_cols} (Indices: {[self.df.columns.get_loc(col) for col in self.reactive_power_cols]})")
        print(
            f"Renewable active power columns: {self.renewable_active_power_cols} (Indices: {[self.df.columns.get_loc(col) for col in self.renewable_active_power_cols]})")
        print(
            f"Renewable reactive power columns: {self.renewable_reactive_power_cols} (Indices: {[self.df.columns.get_loc(col) for col in self.renewable_reactive_power_cols]})")
        print(f"Price columns: {self.price_col} (Indices: {[self.df.columns.get_loc(col) for col in self.price_col]})")
        # Calculate max and min for each type of power
        self.active_power_max = self.df[self.active_power_cols].max().max()
        self.active_power_min = self.df[self.active_power_cols].min().min()

        self.reactive_power_max = self.df[self.reactive_power_cols].max().max() if self.reactive_power_cols else None
        self.reactive_power_min = self.df[self.reactive_power_cols].min().min() if self.reactive_power_cols else None

        self.renewable_active_power_max = self.df[self.renewable_active_power_cols].max().max() \
            if self.renewable_active_power_cols else None
        self.renewable_active_power_min = self.df[self.renewable_active_power_cols].min().min() \
            if self.renewable_active_power_cols else None

        self.renewable_reactive_power_max = self.df[
            self.renewable_reactive_power_cols].max().max() if self.renewable_reactive_power_cols else None
        self.renewable_reactive_power_min = self.df[
            self.renewable_reactive_power_cols].min().min() if self.renewable_reactive_power_cols else None

        self.price_min = self.df[self.price_col].min().values[0] if self.price_col else None
        self.price_max = self.df[self.price_col].max().values[0] if self.price_col else None

        # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        # # price horizen trendency
        # price_all = self.df[self.price_col]
        # # 每天的数据点数
        # points_per_day = 96
        # num_days = len(price_all) // points_per_day
        #
        # # 检查数据是否完整
        # if len(price_all) % points_per_day != 0:
        #     print("Warning: 数据长度不是完整的天数倍数")
        #
        # # 将数据按天分组
        # daily_prices = price_all.values.reshape(-1, points_per_day)
        #
        # # 绘制曲线
        # plt.figure(figsize=(12, 8))
        # for day_idx, daily_price in enumerate(daily_prices):
        #     # plt.plot(range(points_per_day), daily_price, label=f'Day {day_idx + 1}')
        #     plt.plot(range(points_per_day), daily_price)
        #
        # # 图例和轴标签
        # plt.title("Daily Price Variation", fontsize=16)
        # plt.xlabel("Time Interval (15 mins)", fontsize=12)
        # plt.ylabel("Price", fontsize=12)
        # plt.grid(True, linestyle='--', alpha=0.7)
        # # plt.legend(loc='upper right', fontsize=8, ncol=2)  # 调整图例位置和格式
        # plt.tight_layout()
        # plt.show()

        # split the train and test dates
        self.train_dates = []
        self.test_dates = []
        self.limit_test_dates = []
        self.limit_train_dates = []
        self.split_data_set()
        self._replace_nan()
        self._check_for_nan()

    def _replace_nan(self) -> None:
        """
        Replace NaN values in the data with interpolated values or the average of the surrounding values.
        """
        self.df.interpolate(inplace=True)
        # self.df.fillna(method='bfill', inplace=True)
        self.df = self.df.bfill()
        # self.df.fillna(method='ffill', inplace=True)
        self.df = self.df.ffill()

    def _check_for_nan(self) -> None:
        """
        Check if any of the arrays contain NaN values and raise an error if they do
        """
        if self.df.isnull().sum().sum() > 0:
            raise ValueError("Data still contains NaN values after preprocessing")

    def select_timeslot_data(self, year: int, month: int, day: int, timeslot: int) -> np.ndarray:
        """
           Select data for a specific timeslot on a specific day.

           Parameters:
               year (int): The year of the date.
               month (int): The month of the date.
               day (int): The day of the date.
               timeslot (int): The timeslot index.

           Returns:
               np.ndarray: The data for the specified timeslot.
        """
        self.df.index = pd.to_datetime(self.df.index)
        # 判断索引是否已有时区信息
        if self.df.index.tz is None:  # 如果是 tz-naive，则添加 UTC 时区
            self.df.index = self.df.index.tz_localize('UTC')

        dt = (pd.Timestamp(year=year, month=month, day=day, hour=0, minute=0, second=0, tz='UTC') +
              pd.Timedelta(minutes=self.time_interval * timeslot))
        if dt in self.df.index:
            return self.df.loc[dt].values
        # 向后查找最近的节点值
        idx = self.df.index.searchsorted(dt, side='right')
        if idx < len(self.df.index):
            nearest_time = self.df.index[idx]
        else:
            nearest_time = self.df.index[-1]
        return self.df.loc[nearest_time].values

    def select_day_data(self, year: int, month: int, day: int) -> np.ndarray:
        """
        Select data for a specific day.

        Parameters:
            year (int): The year of the date.
            month (int): The month of the date.
            day (int): The day of the date.

        Returns:
            np.ndarray: The data for the specified day.
        """
        start_dt = pd.Timestamp(year=year, month=month, day=day, hour=0, minute=0, second=0, tz='UTC')
        end_dt = start_dt + pd.Timedelta(days=1)
        day_data = self.df.loc[start_dt:end_dt - pd.Timedelta(minutes=1), :]
        return day_data.values

    def list_dates(self) -> List[Tuple[int, int, int]]:
        """
               List all available dates in the data.

               Returns:
                   List[Tuple[int, int, int]]: A list of available dates as (year, month, day).
               """
        dates = self.df.index.strftime('%Y-%m-%d').unique()
        year_month_day = [(int(date[:4]), int(date[5:7]), int(date[8:10])) for date in dates]
        return year_month_day

    def random_date(self) -> Tuple[int, int, int]:
        """
                Randomly select a date from the available dates in the data.

                Returns:
                    Tuple[int, int, int]: The year, month, and day of the selected date.
                """
        dates = self.list_dates()
        year, month, day = random.choice(dates)
        return year, month, day
    def split_data_set(self):
        """
        Split the data into training and testing sets based on the date.

        The first three weeks of each month are used for training and the last week for testing.
        """
        all_dates = self.list_dates()
        all_dates.sort(key=lambda x: (x[0], x[1], x[2]))  # Sort dates

        train_dates = []
        test_dates = []
        # To ensure a long-term test, the last self.long_term_eval_days - 1 days of each month should not be
        # selected to prevent an incomplete evaluation period.
        limit_train_dates = []
        limit_test_dates = []

        current_month = all_dates[0][1]
        current_year = all_dates[0][0]
        monthly_dates = []

        for date in all_dates:
            year, month, day = date
            if month != current_month or year != current_year:
                # Sort monthly dates and split into train and test
                monthly_dates.sort()
                train_len = int(len(monthly_dates) * (3 / 4))  # First three weeks for training
                train_dates += monthly_dates[:train_len]
                test_dates += monthly_dates[train_len:]

                limit_train_dates += monthly_dates[:train_len-self.long_term_eval_days+1]
                limit_test_dates += monthly_dates[train_len:len(monthly_dates)-self.long_term_eval_days+1]

                # Reset for the new month
                monthly_dates = []
                current_month = month
                current_year = year

            monthly_dates.append(date)

        # Handle the last month
        if len(monthly_dates) > 0:
            monthly_dates.sort()
            train_len = int(len(monthly_dates) * (3 / 4))
            train_dates += monthly_dates[:train_len]
            test_dates += monthly_dates[train_len:]
            limit_train_dates += monthly_dates[:train_len - self.long_term_eval_days + 1]
            limit_test_dates += monthly_dates[train_len:len(monthly_dates) - self.long_term_eval_days + 1]


        self.train_dates = train_dates
        self.test_dates = test_dates
        self.limit_train_dates = limit_train_dates
        self.limit_test_dates = limit_test_dates