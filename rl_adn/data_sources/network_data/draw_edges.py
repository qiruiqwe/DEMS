import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置随机种子，确保布局可重复
np.random.seed(42)  # 设置 numpy 的随机种子
nx.algorithms.spring_layout_seed = 42  # 设置 networkx 内部的种子（视版本支持）

# 1. 获取用户输入的节点数
# "请输入节点数（例如 25、34、123）："
node_count = 123

# 2. 动态构造文件路径
folder_name = f"node_{node_count}"
file_name = f"lines_{node_count}.csv"
file_path = os.path.join(folder_name, file_name)

# 3. 检查文件是否存在
if not os.path.exists(file_path):
    print(f"错误：文件 {file_path} 不存在！请确认输入的节点数是否正确。")
    exit(1)

# 4. 读取 CSV 文件
data = pd.read_csv(file_path, header=0)  # 假设 CSV 文件无表头

# 5. 提取节点连接关系（前两列）
edges = data.iloc[:, [0, 1]].values.tolist()  # 提取第 0 列和第 1 列（忽略表头）

# 6. 创建图对象
G = nx.Graph()  # 使用无向图，若需要有向图可改为 nx.DiGraph()

# 7. 添加边
G.add_edges_from(edges)

# 8. 绘制拓扑图
plt.figure(figsize=(16, 10))  # 设置图形大小
# pos = nx.spring_layout(G, seed=42)  # 使用弹簧布局（可替换为其他布局，如 circular_layout）
# pos = nx.circular_layout(G)  # 圆形布局
pos = nx.kamada_kawai_layout(G)

nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=400, font_size=10, edge_color='gray')
plt.title(f"IEEE {node_count} Node System Topology")
plt.show()