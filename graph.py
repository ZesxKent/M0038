# Import necessary libraries
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
from sklearn.utils import resample
from torch import nn
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Load the new CSV file
file_path_new = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot429_GAN_1.csv'
df = pd.read_csv(file_path_new)

# Drop unnecessary columns
df.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault',
                 ], inplace=True)

float_columns = [
    'Inspected_Mass_g', 'PickAttempt_RipenessProbability',
    'FoundBerries_Perspective_(Global,3)', 'FoundBerries_WorkVolumeTotalCount',
    'FoundBerries_WorkVolumeRipeCount', 'FoundBerries_OutsideWorkVolumeTotalCount',
    'FoundBerries_OutsideWorkVolumeRipeCount', 'Levelled_RollDegrees', 'time_normalized'
]
df.fillna(df.median(), inplace=True)
# 找出所有需要转换为整数的列（即不在上述float_columns中的列）
integer_columns = [col for col in df.columns if col not in float_columns]

# 对这些列进行四舍五入并转换为整数
for col in integer_columns:
    df[col] = np.round(df[col]).astype(int)

# 如果有必要，可以对特定的列使用clip方法以确保合理范围
# 例如，如果某些整数列有逻辑上的限制（比如不能为负数），可以使用clip来约束范围
for col in integer_columns:
    df[col] = df[col].clip(lower=0)  # 假设下限是0，根据实际情况调整

# 2. Handle Class Imbalance by Downsampling
df_danger = df[df['Danger_Status'] == 1]
df_normal = df[df['Danger_Status'] == 0]

df_normal_downsampled = resample(df_normal,
                                 replace=False,
                                 n_samples=len(df_danger),
                                 random_state=42)

# Combine the downsampled class 0 and original class 1
df_balanced = pd.concat([df_normal_downsampled, df_danger])

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
df = df_balanced
# 查看处理后的数据集
print(df.head())

# 选取特征列（假设最后一列是目标列）
features_new = df.columns[df.columns != 'Danger_Status']
print(features_new)

# 计算每个特征之间的相关性矩阵
correlation_matrix = df[features_new].corr().abs()  # 绝对值，确保相关性为正值

# 初始化一个无向图
G_correlation = nx.Graph()

# 添加所有特征节点
for feature in features_new:
    G_correlation.add_node(feature)

# 设置相关性阈值，如果相关性超过阈值则添加边
correlation_threshold = 0.02  # 可以根据数据调整阈值

# 根据相关性矩阵添加边，相关性超过阈值的特征对被连接
for i, feature1 in enumerate(features_new):
    for j, feature2 in enumerate(features_new[i+1:]):
        corr_value = correlation_matrix.iloc[i, j + i + 1]  # 获取相关性
        if corr_value > correlation_threshold:
            G_correlation.add_edge(feature1, feature2, weight=corr_value)  # 将相关性作为边的权重

# 打印图的节点和边信息
print(f"Graph with {G_correlation.number_of_nodes()} nodes and {G_correlation.number_of_edges()} edges")

# 计算每个节点的度数，度数为 0 的节点就是孤立节点
isolated_nodes = [node for node, degree in dict(G_correlation.degree()).items() if degree == 0]

# 打印孤立的节点
print(f"Isolated nodes: {isolated_nodes}")

# 从图中删除这些孤立节点
G_correlation.remove_nodes_from(isolated_nodes)

# 打印删除孤立节点后的图信息
print(f"Graph after removing isolated nodes: {G_correlation.number_of_nodes()} nodes and {G_correlation.number_of_edges()} edges")

# 可视化删除孤立节点后的图
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G_correlation, seed=44)  # 为了美观的节点布局
edge_weights = nx.get_edge_attributes(G_correlation, 'weight')  # 获取边权重用于显示
nx.draw(G_correlation, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=3, font_weight="bold", edge_color="gray")
plt.title("Graph After Removing Isolated Nodes")
plt.show()

# 重新创建节点映射：将特征名称映射为整数索引（删除孤立节点后）
node_mapping = {node: i for i, node in enumerate(G_correlation.nodes)}

# 获取边并将边的起点和终点从名称转换为对应的整数索引
edges_mapped = [(node_mapping[u], node_mapping[v]) for u, v in G_correlation.edges()]

# 将边列表转换为 PyTorch tensor，并转置为 [2, num_edges] 的形状
edge_index = torch.tensor(edges_mapped, dtype=torch.long).t().contiguous()

# 节点特征矩阵 (假设每个节点的特征是原始数据表中对应列的值)
# 删除孤立节点后需要重新选择节点对应的特征
x_filtered = df[features_new].loc[:, df[features_new].columns.isin(node_mapping.keys())].values
x = torch.tensor(x_filtered, dtype=torch.float)

# 标签数据（假设最后一列是 Danger_Status，需要分类 0 或 1）
y = torch.tensor(df['Danger_Status'].values, dtype=torch.long)

# 创建图数据对象
data = Data(x=x, edge_index=edge_index, y=y)

# 进行特征归一化
data = NormalizeFeatures()(data)

# 输出数据对象信息
print(f"Graph data after removing isolated nodes: {data}")


import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# 定义 GCN 模型

class DeepGCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(DeepGCNModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))  # 第一层

        # 增加隐藏层
        for _ in range(num_layers - 2):  # 中间层
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, output_dim))  # 最后一层
        self.dropout = dropout

        self.dropout = 0.5  # Dropout 概率设为 0.5

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 应用多层图卷积
        for conv in self.convs[:-1]:  # 对前几层卷积，应用激活函数和 dropout
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 输出层，不需要激活函数
        x = self.convs[-1](x, edge_index)

        return F.log_softmax(x, dim=1)


# 定义 GCN-LSTM 模型
class GCN_LSTM_Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_hidden_dim, output_dim, num_gcn_layers=3, num_lstm_layers=1,
                 dropout=0.5):
        super(GCN_LSTM_Model, self).__init__()

        # GCN部分
        self.gcn_convs = torch.nn.ModuleList()
        self.gcn_convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_gcn_layers - 1):
            self.gcn_convs.append(GCNConv(hidden_dim, hidden_dim))

        # LSTM部分
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, num_lstm_layers, batch_first=True, dropout=dropout)

        # 输出层
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GCN部分：提取空间特征
        for conv in self.gcn_convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
# 模型的输入和输出维度
input_dim = data.x.shape[1]  # 输入特征的维度
hidden_dim = 256  # 隐藏层的维度，可以调节
output_dim = 2  # 二分类问题（0 或 1）
num_layers = 4  # 增加层数

# 初始化模型
model = DeepGCNModel(input_dim, hidden_dim, output_dim, num_layers=num_layers)

print(model)
import torch.optim as optim

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器


# 训练循环
def train(model, data, optimizer, criterion, epochs=400):
    model.train()  # 模型进入训练模式
    for epoch in range(epochs):
        optimizer.zero_grad()  # 梯度清零
        out = model(data)  # 前向传播

        # 计算损失值
        loss = criterion(out, data.y)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 每隔 20 轮输出一次损失值
        if epoch % 20 == 0:
            print(f'Epoch {epoch} - Loss: {loss.item()}')


# 开始训练模型
train(model, data, optimizer, criterion)

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


# 评估模型的准确率和其他性能指标
def evaluate_model(model, data):
    model.eval()  # 模型进入评估模式
    with torch.no_grad():  # 关闭梯度计算
        # 预测类别
        pred = model(data).argmax(dim=1).cpu().numpy()  # 将预测结果转为 numpy 数组
        true_labels = data.y.cpu().numpy()  # 获取真实的标签

        # 计算各种性能指标
        report = classification_report(true_labels, pred, digits=4)
        accuracy = accuracy_score(true_labels, pred)

        # 对于二分类问题，计算 ROC-AUC
        if len(set(true_labels)) == 2:
            prob = model(data).softmax(dim=1)[:, 1].cpu().numpy()  # 获取预测的概率
            roc_auc = roc_auc_score(true_labels, prob)
            print(f'ROC-AUC Score: {roc_auc:.4f}')

        # 打印报告
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", report)


# 评估训练好的模型
evaluate_model(model, data)

# # Extract the features and labels from the DataFrame
# features = df_new[features_new].values
# labels = df_new['Danger_Status'].values  # Assuming 'Danger_Status' is the target column
#
# # Convert the features and labels to torch tensors
# x = torch.tensor(features, dtype=torch.float)
# y = torch.tensor(labels, dtype=torch.long)
#
# # Extract edges from the graph structure
# edges = nx.convert_matrix.to_numpy_array(G_final_new)
# print(edges)
# # Convert the edge list to PyTorch tensors
# edge_index = torch.tensor(np.array(edges.nonzero()), dtype=torch.long)
# print(edge_index)
# # Create a PyTorch Geometric graph object
# data = Data(x=x, edge_index=edge_index, y=y)
#
# # Optionally normalize the features
# data = T.NormalizeFeatures()(data)
#
# # Print the data object
# print(data)

# df_new = df
# # Extract the columns that are features (excluding Danger_Status)
# features_new = df_new.columns[:-1]
#
# # Calculate the feature importance sum for normalizing the values
# feature_sums_new = df_new[features_new].sum()
#
# # Normalize feature values by dividing by the sum
# normalized_weights_new = feature_sums_new / feature_sums_new.sum()
#
# # Initialize a dictionary to store feature connections
# feature_connections_new = {}
#
# # Full implementation with average weight distribution for fully connected edges
# G_final_new = nx.Graph()
#
# # Hub node
# hub_node_new = "Danger_Status"
#
# # Add the hub node "Danger_Status"
# G_final_new.add_node(hub_node_new)
#
# # Add feature nodes and connect them to the hub node
# for feature in normalized_weights_new.index:
#     prefix = feature.split('_')[0]  # Extract the prefix before '_'
#
#     # Add the feature node
#     G_final_new.add_node(feature)
#
#     # Connect the feature to the hub node (Danger_Status) with the normalized weight
#     G_final_new.add_edge(hub_node_new, feature, weight=normalized_weights_new[feature])
#
#     # Store features by their prefixes for later full connection
#     if prefix not in feature_connections_new:
#         feature_connections_new[prefix] = []
#     feature_connections_new[prefix].append(feature)
#
# # Add full connections between features with the same prefix using average weight distribution
# for feature_list in feature_connections_new.values():
#     if len(feature_list) > 1:
#         # Connect each feature in the list to every other feature
#         for i, feature1 in enumerate(feature_list):
#             for feature2 in feature_list[i + 1:]:
#                 # Calculate average weight for the full connection between two features
#                 avg_weight = (normalized_weights_new[feature1] + normalized_weights_new[feature2]) / 2
#                 G_final_new.add_edge(feature1, feature2, weight=avg_weight)
#
# # Remove self-loops (if any) from the graph
# if G_final_new.has_edge(hub_node_new, hub_node_new):
#     G_final_new.remove_edge(hub_node_new, hub_node_new)
#
# print(G_final_new)
#
# # Visualize the final graph without self-loops
# plt.figure(figsize=(10, 10))
# pos = nx.spring_layout(G_final_new, seed=2)  # Positioning of the nodes
# edge_weights_new = nx.get_edge_attributes(G_final_new, 'weight')  # Get edge weights for display
# nx.draw(G_final_new, pos, with_labels=True, node_size=1000, node_color="lightblue", font_size=5, font_weight="bold",
#         edge_color="gray")
# nx.draw_networkx_edge_labels(G_final_new, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_weights_new.items()})
# plt.title("Graph with Average Weight Distribution (Without Self-Loops)")
# plt.show()
#
# import torch
# from torch_geometric.data import Data
# import torch_geometric.transforms as T
#
# # Extract the features and labels from the DataFrame
# features = df_new[features_new].values
# labels = df_new['Danger_Status'].values  # Assuming 'Danger_Status' is the target column
#
# # Convert the features and labels to torch tensors
# x = torch.tensor(features, dtype=torch.float)
# y = torch.tensor(labels, dtype=torch.long)
#
# # Extract edges from the graph structure
# edges = nx.convert_matrix.to_numpy_array(G_final_new)
# print(edges)
# # Convert the edge list to PyTorch tensors
# edge_index = torch.tensor(np.array(edges.nonzero()), dtype=torch.long)
# print(edge_index)
# # Create a PyTorch Geometric graph object
# data = Data(x=x, edge_index=edge_index, y=y)
#
# # Optionally normalize the features
# data = T.NormalizeFeatures()(data)
#
# # Print the data object
# print(data)
#
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
#
#
# # Define a GCN-based model
# class GCNModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GCNModel, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, output_dim)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         # Apply graph convolution layers with ReLU activations
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#
#         return F.log_softmax(x, dim=1)
#
#
# # Define input, hidden, and output dimensions
# input_dim = x.shape[1]  # Number of features
# hidden_dim = 64  # Can be tuned
# output_dim = 2  # Binary classification (0 or 1)
#
# # Initialize the model
# model = GCNModel(input_dim, hidden_dim, output_dim)
#
# import torch.optim as optim
#
# # Define optimizer and loss function
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# criterion = nn.CrossEntropyLoss()
#
#
# # Training loop
# def train(model, data, optimizer, criterion, epochs=200):
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         out = model(data)
#
#         # Only use the Danger_Status node for loss calculation
#         loss = criterion(out[data.y == 1], data.y[data.y == 1])
#
#         loss.backward()
#         optimizer.step()
#
#         if epoch % 20 == 0:
#             print(f'Epoch {epoch} - Loss: {loss.item()}')
#
#
# # Train the model
# train(model, data, optimizer, criterion)
#
# # Rename the test function to avoid pytest conflict
# def evaluate_model(model, data):
#     model.eval()
#     with torch.no_grad():
#         pred = model(data).argmax(dim=1)
#         correct = (pred == data.y).sum().item()
#         accuracy = correct / len(data.y)
#         print(f'Accuracy: {accuracy:.4f}')
#
# # Evaluate the model after training
# evaluate_model(model, data)

