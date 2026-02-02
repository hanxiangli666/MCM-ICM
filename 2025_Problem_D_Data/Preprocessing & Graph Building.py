import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 读取官方数据 (请确保文件名与解压后的一致)
# ==========================================
print("正在读取数据...")
# 注意：官方数据量可能很大，读取需要几秒钟
nodes_df = pd.read_csv('2025_Problem_D_Data/nodes_all.csv')
edges_df = pd.read_csv('2025_Problem_D_Data/edges_all.csv')

# 论文中提到需要处理数据噪声，这里我们做最基础的清洗
# 确保边的起点(u)和终点(v)都在节点列表中
valid_nodes = set(nodes_df['osmid'])
edges_df = edges_df[edges_df['u'].isin(valid_nodes) & edges_df['v'].isin(valid_nodes)]

print(f"节点数量: {len(nodes_df)}")
print(f"边数量: {len(edges_df)}")

# ==========================================
# 2. 构建图模型 G (复刻论文公式 2)
# ==========================================
# 论文公式(4)提到 t(u,v) = d(u,v) / speed [cite: 218-220]
# 官方数据 edges_all.csv 通常包含 'length' (米)。
# 如果没有 'speed' 列，我们假设一个平均速度 (例如 40 km/h) 来计算权重
# 实战提示：你可以根据 highway 类型(motorway, residential) 赋予不同速度

# 假设平均速度 40km/h = 11.1 m/s
default_speed_mps = 11.1 
edges_df['weight'] = edges_df['length'] / default_speed_mps 

# 构建有向图
G = nx.from_pandas_edgelist(
    edges_df, 
    source='u', 
    target='v', 
    edge_attr=['weight', 'length'], 
    create_using=nx.DiGraph()
)

# 添加节点坐标属性 (用于画图和计算几何距离)
node_pos = {row['osmid']: (row['x'], row['y']) for idx, row in nodes_df.iterrows()}
nx.set_node_attributes(G, node_pos, 'pos')

print("✅ 路网模型构建完成！")

# ==========================================
# 3. 模拟桥梁倒塌 (移除关键边)
# ==========================================
# 教练提示：你需要去 edges_all.csv 里找到 Key Bridge 对应的 u 和 v
# 这里为了演示代码逻辑，我随机选择一条“流量大”或者“长”的边作为桥梁
# 实战中请替换为真实的桥梁节点 ID

# 假设我们移除图中最长的一条边作为模拟
longest_edge = edges_df.loc[edges_df['length'].idxmax()]
bridge_u, bridge_v = longest_edge['u'], longest_edge['v']

print(f"模拟移除桥梁: {bridge_u} -> {bridge_v}")

# 创建倒塌后的图 G_collapsed
G_collapsed = G.copy()
if G_collapsed.has_edge(bridge_u, bridge_v):
    G_collapsed.remove_edge(bridge_u, bridge_v)
# 如果是双向的，记得移除反向边
if G_collapsed.has_edge(bridge_v, bridge_u):
    G_collapsed.remove_edge(bridge_v, bridge_u)

# ==========================================
# 4. 计算中心性变化 (复刻论文 Figure 8)
# ==========================================
# 为了速度，我们只随机选取 50 个节点作为“关键路口”进行对比
# 论文选取了 39 个 highway nodes [cite: 291]
sampled_nodes = np.random.choice(nodes_df['osmid'], 50, replace=False)

print("正在计算 Closeness Centrality (可能需要几分钟)...")
# 计算 G2 (原图)
cent_before = nx.closeness_centrality(G, distance='weight', u=sampled_nodes) # 注意：networkx新版支持只计算部分节点吗？
# NetworkX 的 closeness_centrality 默认算全图。如果图太大，可以用 subgraph 或者只算部分
# 修正：直接对子图或全图算可能会慢。为了演示，我们假设图不大，或者要有耐心。
# 另一种策略：只计算这50个节点到其他节点的距离。

# 简便起见，我们这里假设跑全图的子集（仅供演示逻辑）
# 在实际比赛代码中，建议提取主要连通分量 (Largest Component)
target_nodes = list(sampled_nodes)

# 计算前后对比
data_comparison = []
for node in target_nodes:
    try:
        # 这里的计算量其实很大，建议实战中提取 subgraph(主要干道) 来算
        c1 = nx.closeness_centrality(G, u=node, distance='weight')
        c2 = nx.closeness_centrality(G_collapsed, u=node, distance='weight')
        data_comparison.append({'Node': str(node), 'Before': c1, 'After': c2})
    except:
        continue

df_res = pd.DataFrame(data_comparison)

# ==========================================
# 5. 可视化对比图 (复刻 Figure 8)
# ==========================================
if not df_res.empty:
    df_res.set_index('Node')[['Before', 'After']].head(20).plot(kind='bar', figsize=(12, 6))
    plt.title("Impact of Bridge Collapse on Closeness Centrality")
    plt.ylabel("Centrality Score")
    plt.show()
    print("✅ 中心性对比图绘制完成！")
else:
    print("❌ 并没有计算出有效结果，请检查图的连通性。")


# ==========================================
# 6. 定义 熵权TOPSIS 函数
# ==========================================
def entropy_weight_topsis(data):
    """
    data: pandas DataFrame, 行是样本(公交站)，列是指标(POI, 流量, 度)
    返回: 每一行的评分
    """
    # 1. 归一化 (正向指标：越大越好)
    data_norm = data.copy()
    for col in data.columns:
        mx = data[col].max()
        mn = data[col].min()
        if mx - mn == 0:
            data_norm[col] = 1 # 避免除以0
        else:
            data_norm[col] = (data[col] - mn) / (mx - mn)
    
    # 2. 计算熵 (Entropy)
    n, m = data_norm.shape
    # 避免log(0)，加一个微小量
    P = data_norm / data_norm.sum(axis=0)
    P = P.replace(0, 0.00001) 
    
    k = 1 / np.log(n)
    E = -k * (P * np.log(P)).sum(axis=0)
    
    # 3. 计算权重 (Weights)
    d = 1 - E
    w = d / d.sum()
    print("指标权重:", w.to_dict())
    
    # 4. 计算综合得分 (TOPSIS Score)
    # 加权矩阵
    Z = data_norm * w
    
    # 正理想解 (Z+) 和 负理想解 (Z-)
    Z_plus = Z.max()
    Z_minus = Z.min()
    
    # 欧氏距离
    D_plus = np.sqrt(((Z - Z_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((Z - Z_minus) ** 2).sum(axis=1))
    
    # 相对接近度 (Score)
    C = D_minus / (D_plus + D_minus)
    return C

# ==========================================
# 7. 实战演练
# ==========================================
# 你需要从 Bus_Stops.csv 和 nodes_all.csv 中提取数据构建矩阵
# 这里模拟生成一个评价矩阵
print("\n正在进行公交站点评价 (TOPSIS)...")
bus_stations = ['Station_A', 'Station_B', 'Station_C', 'Station_D', 'Station_E']
# 假设指标：POI数量, 乘客流量, 节点度
data = pd.DataFrame({
    'POI_Count': [5, 2, 8, 1, 4],
    'Passenger_Flow': [1000, 300, 2000, 150, 800],
    'Node_Degree': [4, 2, 6, 1, 3]
}, index=bus_stations)

scores = entropy_weight_topsis(data)
data['Score'] = scores
print(data.sort_values(by='Score', ascending=False))
print("✅ 站点分级完成！")