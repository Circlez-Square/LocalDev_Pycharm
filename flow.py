import networkx as nx
import matplotlib.pyplot as plt

# 创建有向图
G = nx.DiGraph()

# 添加节点
nodes = [
    "Load Raw Data",
    "Remove Invalid Dates",
    "Remove Missing and Duplicate Values",
    "Remove Predictive Columns Retaining Valid Data",
    "Remove Invalid Fundraising Goals",
    "Remove Invalid Statuses",
    "Data with Invalid Statuses",
    "New Data"
]

G.add_nodes_from(nodes)

# 添加边
edges = [
    ("Load Raw Data", "Remove Invalid Dates"),
    ("Remove Invalid Dates", "Remove Missing and Duplicate Values"),
    ("Remove Missing and Duplicate Values", "Remove Predictive Columns Retaining Valid Data"),
    ("Remove Predictive Columns Retaining Valid Data", "Remove Invalid Fundraising Goals"),
    ("Remove Predictive Columns Retaining Valid Data", "Remove Invalid Statuses"),
    ("Remove Invalid Fundraising Goals", "Data with Invalid Statuses"),
    ("Remove Invalid Statuses", "New Data")
]

G.add_edges_from(edges)

# 设置节点和边的样式
plt.figure(figsize=(14, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=6000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True, edge_color='gray', linewidths=1, font_color='black')
plt.title("Directed Acyclic Graph (DAG)", fontsize=15)
plt.show()
