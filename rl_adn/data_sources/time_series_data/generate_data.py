import pandas as pd
import numpy as np
from scipy.stats import norm

# 1. 加载已知数据
data_25 = pd.read_csv("25_node_time_series.csv")

# 2. 分析可再生能源有功功率的分布特征
def analyze_renewable_distribution(data):
    """分析可再生能源有功功率的分布特征"""
    renewable_columns = [col for col in data.columns if col.startswith("renewable_active_power_node")]
    renewable_values = data[renewable_columns].values.flatten()
    # 只分析发电量大于0的数据
    renewable_values = renewable_values[renewable_values > 0]
    mean = np.mean(renewable_values)
    std = np.std(renewable_values)
    return mean, std

renewable_mean, renewable_std = analyze_renewable_distribution(data_25)

# 3. 生成6个月的时间戳
start_date = "2021-01-01 00:00"
end_date = "2021-06-30 23:45"
date_range = pd.date_range(start=start_date, end=end_date, freq="15T")
num_time_points = len(date_range)
num_nodes = 123

# 4. 生成可再生能源有功功率数据
renewable_active_power_data = np.zeros((num_time_points, num_nodes))

for i, timestamp in enumerate(date_range):
    hour = timestamp.hour
    # 假设白天（6:00到18:00）有发电
    if 5 <= hour < 17:
        # 生成符合正态分布的可再生能源有功功率数据
        renewable_active_power_data[i, :] = np.random.normal(loc=renewable_mean, scale=renewable_std, size=num_nodes)
        # 确保发电量不为负数
        renewable_active_power_data[i, :] = np.maximum(renewable_active_power_data[i, :], 0)

# 5. 生成有功功率数据（基于已知数据的分布特征）
active_power_mean, active_power_std = np.mean(data_25[[col for col in data_25.columns if col.startswith("active_power_node")]].values), \
                                      np.std(data_25[[col for col in data_25.columns if col.startswith("active_power_node")]].values)
active_power_data = np.random.normal(loc=active_power_mean, scale=active_power_std, size=(num_time_points, num_nodes))
active_power_columns = [f"active_power_node_{i+1}" for i in range(num_nodes)]

# 6. 生成电价数据（基于已知数据的分布特征）
price_mean, price_std = np.mean(data_25["price"]), np.std(data_25["price"])
price_data = np.random.normal(loc=price_mean, scale=price_std, size=num_time_points)

# 7. 构建DataFrame
data = {
    "date_time": date_range
}
data.update({col: active_power_data[:, i] for i, col in enumerate(active_power_columns)})
data.update({col: renewable_active_power_data[:, i] for i, col in enumerate([f"renewable_active_power_node_{i+1}" for i in range(num_nodes)])})
data["price"] = price_data

df = pd.DataFrame(data)

# 8. 保存为CSV文件
df.to_csv("123_node_time_series.csv", index=False)

print("123_node_time_series.csv 文件已生成！")