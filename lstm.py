# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
#
# # 1. Load and Prepare Data
# file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\combined_data_without_inspected_mass_g.csv'
# df = pd.read_csv(file_path)
#
# # Sort by time to ensure temporal order
# df = df.sort_values(by='time').reset_index(drop=True)
#
#
# # Drop unnecessary columns
# df.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault',
#                  'CompletedTrundle_Displacement_mm', 'Event_Supervisor.InspectionFault'], inplace=True)
#
# # Feature scaling
# scaler = MinMaxScaler()
# X = df.drop(columns=['Danger_Status', 'time', 'Unnamed: 0'])
# y = df['Danger_Status']
# X_scaled = scaler.fit_transform(X)
#
# # 2. Reshape data for LSTM with time order
# timesteps = 100  # Define the number of time steps to use for each sample
# X_lstm = []
# y_lstm = []
#
# for i in range(len(X_scaled) - timesteps):
#     X_lstm.append(X_scaled[i:i + timesteps])
#     y_lstm.append(y[i + timesteps])
#
# X_lstm = np.array(X_lstm)
# y_lstm = np.array(y_lstm)
#
# # Convert to PyTorch tensors
# X_lstm = torch.tensor(X_lstm, dtype=torch.float32)
# y_lstm = torch.tensor(y_lstm, dtype=torch.float32)
#
# # 3. Split Data based on time
# split_index = int(len(X_lstm) * 0.8)  # Use 80% of the data for training
# X_train, X_test = X_lstm[:split_index], X_lstm[split_index:]
# y_train, y_test = y_lstm[:split_index], y_lstm[split_index:]
#
#
# # 4. Define the LSTM Model in PyTorch
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         out = self.sigmoid(out)
#         return out
#
#
# # 5. Initialize the Model, Loss Function, and Optimizer
# input_size = X_train.shape[2]
# hidden_size = 50
# num_layers = 2
# learning_rate = 0.001
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = LSTMModel(input_size, hidden_size, num_layers).to(device)
#
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # 6. Train the Model
# num_epochs = 50
# batch_size = 32
#
# X_train = X_train.to(device)
# y_train = y_train.to(device)
# X_test = X_test.to(device)
# y_test = y_test.to(device)
#
# for epoch in range(num_epochs):
#     model.train()
#     permutation = torch.randperm(X_train.size(0))
#     for i in range(0, X_train.size(0), batch_size):
#         indices = permutation[i:i + batch_size]
#         batch_x, batch_y = X_train[indices], y_train[indices]
#
#         outputs = model(batch_x).squeeze()
#         loss = criterion(outputs, batch_y)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     if (epoch + 1) == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# # 7. Evaluate the Model
# model.eval()
# with torch.no_grad():
#     y_pred_prob = model(X_test).cpu().numpy()
#     y_pred = (y_pred_prob > 0.5).astype(int)
#
#     accuracy = accuracy_score(y_test.cpu(), y_pred)
#     roc_auc = roc_auc_score(y_test.cpu(), y_pred_prob)
#     report = classification_report(y_test.cpu(), y_pred)
#
# print(f"Accuracy: {accuracy}")
# print(f"ROC-AUC Score: {roc_auc}")
# print("Classification Report:\n", report)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load and Prepare Data
file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot429_LGAN_test1.csv'
df = pd.read_csv(file_path)

# 删除不必要的列
df.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault',
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
    df[col] = df[col].clip(lower=0)

# 加载新数据并预处理
new_file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot429_pro.csv'
df_1 = pd.read_csv(new_file_path)
df_1.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault',
                   'time','time_in_seconds'], inplace=True)

# 删除 df_2 中 Danger_Status 为 0 的行
df_2_filtered = df[df['Danger_Status'] != 0]

# 合并 df_1 和过滤后的 df_2
df = pd.concat([df_1, df_2_filtered], ignore_index=True)
df = df.sort_values(by='time_normalized', ascending=True)
# 查看合并后的数据
print(df)

# 将数据划分为序列，确保其大小是时间步的整数倍
time_steps = 20  # 时间步
feature_dim = len(df.columns) - 1  # 特征数

# 2. Handle Class Imbalance by Downsampling
df_danger = df[df['Danger_Status'] == 1]
df_normal = df[df['Danger_Status'] == 0]

# 下采样 class 0 以匹配 class 1 的大小
df_normal_downsampled = resample(df_normal, replace=False, n_samples=len(df_danger) * 2, random_state=42)

# 合并下采样后的 class 0 和原始 class 1
df_balanced = pd.concat([df_normal_downsampled, df_danger])

# # 打乱数据集
df_balanced = df_balanced.sample(frac=1, random_state=1).reset_index(drop=True)

df_balanced = df_balanced.sort_values(by='time_normalized', ascending=True)

df_balanced.to_csv('balanced.csv', index=False)
# 3. Feature Scaling
scaler = MinMaxScaler()

# 取出特征矩阵和标签
x_filtered = df_balanced.drop(columns=['Danger_Status']).values  # 删除目标列
y = df_balanced['Danger_Status'].values

# 确保样本总数是 time_steps 的整数倍
n_samples = x_filtered.shape[0]
n_samples = (n_samples // time_steps) * time_steps  # 使其可以被 time_steps 整除
x_filtered = x_filtered[:n_samples]
y = y[:n_samples]

# 计算新的批次大小
batch_size = n_samples // time_steps

# 重塑 x 为 (batch_size, time_steps, feature_dim)
x_reshaped = np.reshape(x_filtered, (batch_size, time_steps, feature_dim))
y_reshaped = np.reshape(y, (batch_size, time_steps))

# 将数据转换为 tensor 并移动到 GPU
x = torch.tensor(x_reshaped, dtype=torch.float).to(device)
y = torch.tensor(y_reshaped[:, -1], dtype=torch.long).to(device)  # 只使用最后一个时间步的标签

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM输入是 (batch_size, time_steps, input_dim)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(self.dropout(lstm_out))  # 全连接层
        return F.log_softmax(out, dim=1)

# 模型参数
input_dim = feature_dim
hidden_dim = 128
output_dim = 2  # 二分类问题
num_layers = 1
dropout = 0.8

# 初始化 LSTM 模型并移动到 GPU 上
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout).to(device)

print(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
def train(model, x, y, optimizer, criterion, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x)  # 前向传播
        loss = criterion(out, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if epoch % 20 == 0:
            print(f'Epoch {epoch} - Loss: {loss.item()}')

# 开始训练模型
train(model, x, y, optimizer, criterion)

# 评估模型
def evaluate_model(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1).cpu().numpy()
        true_labels = y.cpu().numpy()
        report = classification_report(true_labels, pred, digits=4)
        accuracy = accuracy_score(true_labels, pred)

        if len(set(true_labels)) == 2:
            prob = model(x).softmax(dim=1)[:, 1].cpu().numpy()
            roc_auc = roc_auc_score(true_labels, prob)
            print(f'ROC-AUC Score: {roc_auc:.4f}')

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", report)

# 评估训练好的模型
evaluate_model(model, x, y)

# 训练后的模型保存
torch.save(model.state_dict(), 'lstm_model.pth')
print("模型已保存。")

# 加载保存的模型
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout)
model.load_state_dict(torch.load('lstm_model.pth'))
model = model.to(device)
print("模型已加载。")

# 加载新数据集
new_df = pd.read_csv('C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot429_pro.csv')

# 删除不必要的列
new_df.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault',
                     'time','time_in_seconds'], inplace=True)

# 将数据分为正样本（Danger_Status=1）和负样本（Danger_Status=0）
df_danger_new = new_df[new_df['Danger_Status'] == 1]
df_normal_new = new_df[new_df['Danger_Status'] == 0]

# 下采样负样本，保持原有顺序
df_normal_downsampled_new = resample(df_normal_new, replace=False, n_samples=len(df_danger_new) * 2, random_state=42)

# 合并下采样后的负样本和正样本，保持顺序
df_balanced_new = pd.concat([df_normal_downsampled_new, df_danger_new]).sort_index()
df_balanced_new = df_balanced_new.sample(frac=1).reset_index(drop=True)
# 重置索引
df_balanced_new = df_balanced_new.reset_index(drop=True)
df_balanced_new = df_balanced_new.sort_values(by='time_normalized', ascending=True)
print(df_balanced_new)
# 提取特征并处理缺失值（与训练集处理方式一致）
x_new_filtered = df_balanced_new.drop(columns=['Danger_Status']).apply(pd.to_numeric, errors='coerce').fillna(0).values
y_new = df_balanced_new['Danger_Status'].values

# 确保数据类型为 float32
x_new_filtered = np.asarray(x_new_filtered, dtype=np.float32)

# 确保 LSTM 的权重是连续的，以消除 CUDA 的警告
model.lstm.flatten_parameters()

# 确定特征数量
feature_dim = 34  # 每个时间步的特征数
time_steps = 20   # 时间步数

# 获取新数据的总样本大小
n_samples_new = x_new_filtered.shape[0]  # 例如 1067220
print(n_samples_new)
# 确保样本总数是 time_steps 的整数倍
n_valid_samples = (n_samples_new // (time_steps)) * time_steps  # 保证总样本数可以被 time_steps 整除
# 截断数据到有效样本数
x_new_filtered = x_new_filtered[:n_valid_samples * feature_dim]  # 保证数据长度是特征维度的倍数
y_new = y_new[:n_valid_samples]  # 目标列数据
# 移除最后8行
x_new_filtered = x_new_filtered[:-4]
y_new = y_new[:-4]
print(len(x_new_filtered))
print(y_new)
# 重新计算 batch_size_new
batch_size_new = n_valid_samples // time_steps  # 计算 batch_size

# 重塑新数据为 (batch_size_new, time_steps, feature_dim)
x_new_reshaped = np.reshape(x_new_filtered, (batch_size_new, time_steps, feature_dim))

# 将新数据转换为 tensor 并移动到 GPU
x_new_time_series = torch.tensor(x_new_reshaped, dtype=torch.float).to(device)
y_new = torch.tensor(y_new[:batch_size_new], dtype=torch.long).to(device)
print(f"x_new_time_series shape: {x_new_time_series.shape}")
print(f"y_new shape: {y_new.shape}")
# 使用模型对新数据进行评估
evaluate_model(model, x_new_time_series, y_new)

