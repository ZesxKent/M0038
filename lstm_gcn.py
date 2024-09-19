import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
from sklearn.utils import resample
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load and Prepare Data
file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot429_GAN_4.csv'
df = pd.read_csv(file_path)

# Drop unnecessary columns
df.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault'
                 ], inplace=True)

float_columns = [
    'Inspected_Mass_g', 'PickAttempt_RipenessProbability',
    'FoundBerries_Perspective_(Global,3)', 'FoundBerries_WorkVolumeTotalCount',
    'FoundBerries_WorkVolumeRipeCount', 'FoundBerries_OutsideWorkVolumeTotalCount',
    'FoundBerries_OutsideWorkVolumeRipeCount', 'Levelled_RollDegrees'
]
df.fillna(df.median(), inplace=True)
# 找出所有需要转换为整数的列（即不在上述float_columns中的列）
integer_columns = [col for col in df.columns if col not in float_columns]

# 对这些列进行四舍五入并转换为整数
for col in integer_columns:
    df[col] = np.round(df[col]).astype(int)

# 如果有必要，可以对特定的列使用clip方法以确保合理范围

for col in integer_columns:
    df[col] = df[col].clip(lower=0)  # 假设下限是0，根据实际情况调整

new_file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\merged_dataset.csv'
df_1 = pd.read_csv(new_file_path)

# Preprocess new data (ensure consistent processing with training data)
df_1.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault',
                   'time', 'time_in_seconds', 'time_normalized'], inplace=True)

# 删除 df_2 中 Danger_Status 为 0 的行
df_2_filtered = df[df['Danger_Status'] != 0]

# 合并 df_1 和过滤后的 df_2
df = pd.concat([df_1, df_2_filtered], ignore_index=True)

# 查看合并后的数据
print(df)

# 处理类别不平衡问题
df_danger = df[df['Danger_Status'] == 1]
df_normal = df[df['Danger_Status'] == 0]
df_normal_downsampled = resample(df_normal, replace=False, n_samples=len(df_danger), random_state=42)
df_balanced = pd.concat([df_normal_downsampled, df_danger]).sample(frac=1, random_state=42).reset_index(drop=True)
df = df_balanced

# 选取特征列
features_new = df.columns[df.columns != 'Danger_Status']

# 计算相关性矩阵并构建图
correlation_matrix = df[features_new].corr().abs()
G_correlation = nx.Graph()
for feature in features_new:
    G_correlation.add_node(feature)
correlation_threshold = 0.02
for i, feature1 in enumerate(features_new):
    for j, feature2 in enumerate(features_new[i + 1:]):
        corr_value = correlation_matrix.iloc[i, j + i + 1]
        if corr_value > correlation_threshold:
            G_correlation.add_edge(feature1, feature2, weight=corr_value)

# 删除孤立节点
isolated_nodes = [node for node, degree in dict(G_correlation.degree()).items() if degree == 0]
G_correlation.remove_nodes_from(isolated_nodes)

# 节点映射和边
node_mapping = {node: i for i, node in enumerate(G_correlation.nodes)}
edges_mapped = [(node_mapping[u], node_mapping[v]) for u, v in G_correlation.edges()]
edge_index = torch.tensor(edges_mapped, dtype=torch.long).t().contiguous()

# 节点特征矩阵和标签
x_filtered = df[features_new].loc[:, df[features_new].columns.isin(node_mapping.keys())].values
x = torch.tensor(x_filtered, dtype=torch.float)
y = torch.tensor(df['Danger_Status'].values, dtype=torch.long)

# 创建图数据对象
data = Data(x=x, edge_index=edge_index, y=y)
data = NormalizeFeatures()(data)

# 定义 GCN-LSTM 模型
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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

    def forward(self, data, time_steps=5):
        x, edge_index = data.x, data.edge_index

        # GCN部分：提取每个时间步的空间特征
        all_time_steps = []
        for t in range(time_steps):
            x_t = x  # 假设每个时间步有不同特征，可以用 data.x[t] 来选择对应时间步的特征
            for conv in self.gcn_convs:
                x_t = conv(x_t, edge_index)
                x_t = F.relu(x_t)
                x_t = F.dropout(x_t, p=self.dropout, training=self.training)
            all_time_steps.append(x_t)

        # 将所有时间步的特征拼接为 (num_nodes, time_steps, hidden_dim)
        x_time_series = torch.stack(all_time_steps, dim=1)  # (num_nodes, time_steps, hidden_dim)

        # LSTM部分：处理时间序列特征
        lstm_out, _ = self.lstm(x_time_series)  # LSTM输入的是 (batch_size=num_nodes, time_steps, hidden_dim)

        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]

        # 全连接层输出
        out = self.fc(lstm_out)

        return F.log_softmax(out, dim=1)


# 模型参数
input_dim = data.x.shape[1]
hidden_dim = 256  # GCN隐藏层的维度
lstm_hidden_dim = 128  # LSTM隐藏层维度
output_dim = 2  # 二分类问题
num_gcn_layers = 3  # GCN层数
num_lstm_layers = 1  # LSTM层数

# 初始化 GCN-LSTM 模型
model = GCN_LSTM_Model(input_dim, hidden_dim, lstm_hidden_dim, output_dim, num_gcn_layers=num_gcn_layers,
                       num_lstm_layers=num_lstm_layers)

# 计算类别权重
num_positive = (df['Danger_Status'] == 1).sum()  # 正类样本数量
num_negative = (df['Danger_Status'] == 0).sum()  # 负类样本数量

# 计算类别权重，正类（1）的权重会更大，负类（0）的权重会更小
class_weights = torch.tensor([1.0, num_negative / num_positive], dtype=torch.float).to(device)

# 使用加权交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 训练循环
def train(model, data, optimizer, criterion, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch} - Loss: {loss.item()}')


# 开始训练模型
train(model, data, optimizer, criterion)

# 评估模型
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


# 评估函数，增加阈值调整
def evaluate_model_with_threshold(model, data, threshold):  # 修改阈值
    model.eval()
    with torch.no_grad():
        # 获取预测的概率
        prob = model(data).softmax(dim=1)[:, 1].cpu().numpy()

        # 根据阈值进行分类
        pred = (prob >= threshold).astype(int)  # 应用新的阈值
        true_labels = data.y.cpu().numpy()

        # 计算各种性能指标
        report = classification_report(true_labels, pred, digits=4)
        accuracy = accuracy_score(true_labels, pred)
        roc_auc = roc_auc_score(true_labels, prob)

        print(f'ROC-AUC Score: {roc_auc:.4f}')
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", report)


# 评估训练好的模型，并设置不同的分类阈值
evaluate_model_with_threshold(model, data, threshold=0.38)  # 设置阈值为 0.3

# 保存模型的参数
torch.save(model.state_dict(), 'gcn_lstm_model.pth')
print("Model saved successfully!")

# 加载模型
model = GCN_LSTM_Model(input_dim, hidden_dim, lstm_hidden_dim, output_dim, num_gcn_layers=num_gcn_layers,
                       num_lstm_layers=num_lstm_layers)
model.load_state_dict(torch.load('gcn_lstm_model.pth'))
model.to(device)
model.eval()  # 切换为评估模式

print("Model loaded successfully!")

# 加载新的数据集
new_data_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\merged_dataset.csv'
df_new = pd.read_csv(new_data_path)

# 对新的数据集进行与训练数据相同的预处理
# 删除不必要的列
df_new.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault',
                     'time','time_normalized', 'time_in_seconds'], inplace=True)
# 其他预处理步骤（如处理缺失值、标准化、四舍五入、clip等）
# ...（根据训练数据的预处理流程进行）
# 处理类别不平衡问题
df_danger = df_new[df_new['Danger_Status'] == 1]
df_normal = df_new[df_new['Danger_Status'] == 0]
df_normal_downsampled = resample(df_normal, replace=False, n_samples=len(df_danger), random_state=42)
df_balanced = pd.concat([df_normal_downsampled, df_danger]).sample(frac=1, random_state=42).reset_index(drop=True)
df_new = df_balanced

# 构建图结构
features_new = df_new.columns[df_new.columns != 'Danger_Status']  # 使用新数据的特征列
correlation_matrix_new = df_new[features_new].corr().abs()
G_new = nx.Graph()
for feature in features_new:
    G_new.add_node(feature)
correlation_threshold = 0.02
for i, feature1 in enumerate(features_new):
    for j, feature2 in enumerate(features_new[i + 1:]):
        corr_value = correlation_matrix_new.iloc[i, j + i + 1]
        if corr_value > correlation_threshold:
            G_new.add_edge(feature1, feature2, weight=corr_value)

# 删除孤立节点
isolated_nodes = [node for node, degree in dict(G_new.degree()).items() if degree == 0]
G_new.remove_nodes_from(isolated_nodes)

# 构建新的图数据对象
node_mapping_new = {node: i for i, node in enumerate(G_new.nodes)}
edges_mapped_new = [(node_mapping_new[u], node_mapping_new[v]) for u, v in G_new.edges()]
edge_index_new = torch.tensor(edges_mapped_new, dtype=torch.long).t().contiguous()

x_filtered_new = df_new[features_new].loc[:, df_new[features_new].columns.isin(node_mapping_new.keys())].values
x_new = torch.tensor(x_filtered_new, dtype=torch.float)
y_new = torch.tensor(df_new['Danger_Status'].values, dtype=torch.long)  # 如果新的数据集有标签

# 创建图数据对象
new_data = Data(x=x_new, edge_index=edge_index_new, y=y_new)
new_data = NormalizeFeatures()(new_data)
new_data = new_data.to(device)

# 在新数据集上进行预测
def predict_on_new_data(model, data):
    model.eval()
    with torch.no_grad():
        # 获取模型的输出（预测概率）
        prob = model(data).softmax(dim=1)[:, 1].cpu().numpy()  # 获取正类的概率
        pred = (prob >= 0.5).astype(int)  # 使用默认阈值 0.5 进行分类

        return pred, prob



# 对新的数据集进行预测
predictions, probabilities = predict_on_new_data(model, new_data)

# 输出预测结果
print("Predictions: ", predictions)
print("Probabilities: ", probabilities)

# 如果新的数据集有真实标签，可以计算模型的性能
if 'Danger_Status' in df_new.columns:
    from sklearn.metrics import classification_report, accuracy_score

    print("Accuracy: ", accuracy_score(y_new.cpu().numpy(), predictions))
    print("Classification Report:\n", classification_report(y_new.cpu().numpy(), predictions))
