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
time_steps = 40  # 时间步
feature_dim = len(df.columns) - 1  # 特征数

# 2. Handle Class Imbalance by Downsampling
df_danger = df[df['Danger_Status'] == 1]
df_normal = df[df['Danger_Status'] == 0]

# 下采样 class 0 以匹配 class 1 的大小
df_normal_downsampled = resample(df_normal, replace=False, n_samples=len(df_danger), random_state=42)

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

# 定义 MLP 模型
# 定义 MLP 模型
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层全连接层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 第二层全连接层
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 输出层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # 将输入展平，使用 reshape 替代 view
        x = F.relu(self.fc1(x))  # 第一层全连接层 + 激活函数
        x = self.dropout(x)  # Dropout 层，防止过拟合
        x = F.relu(self.fc2(x))  # 第二层全连接层 + 激活函数
        x = self.dropout(x)  # Dropout 层
        out = self.fc3(x)  # 输出层
        return F.log_softmax(out, dim=1)  # 使用 log_softmax 输出预测概率


# 模型参数
input_dim = feature_dim * time_steps  # MLP的输入是展平的，即原来时间步和特征维度的乘积
hidden_dim = 128
output_dim = 2  # 二分类问题
dropout = 0.5

# 初始化 MLP 模型并移动到 GPU 上
model = MLPModel(input_dim, hidden_dim, output_dim, dropout=dropout).to(device)
print(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
torch.save(model.state_dict(), 'mlp_model.pth')
print("模型已保存。")

# 加载保存的模型
model = MLPModel(input_dim, hidden_dim, output_dim, dropout=dropout)
model.load_state_dict(torch.load('mlp_model.pth'))
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
df_normal_downsampled_new = resample(df_normal_new, replace=False, n_samples=len(df_danger_new), random_state=42)

# 合并下采样后的负样本和正样本，保持顺序
df_balanced_new = pd.concat([df_normal_downsampled_new, df_danger_new]).sort_index()
df_balanced_new = df_balanced_new.sample(frac=1).reset_index(drop=True)

# 提取特征并处理缺失值（与训练集处理方式一致）
x_new_filtered = df_balanced_new.drop(columns=['Danger_Status']).apply(pd.to_numeric, errors='coerce').fillna(0).values
y_new = df_balanced_new['Danger_Status'].values

# 确保数据类型为 float32
x_new_filtered = np.asarray(x_new_filtered, dtype=np.float32)

# 计算新数据集的样本总数
n_samples_new = x_new_filtered.shape[0]

# 确保样本总数是 time_steps 的整数倍
n_valid_samples = (n_samples_new // time_steps) * time_steps
x_new_filtered = x_new_filtered[:n_valid_samples]  # 保证输入数据与 n_valid_samples 对齐
y_new = y_new[:n_valid_samples]  # 保证标签与输入数据对齐

# 计算新的批次大小
batch_size_new = n_valid_samples // time_steps

# 重塑新数据为 (batch_size_new, time_steps, feature_dim)
x_new_reshaped = np.reshape(x_new_filtered, (batch_size_new, time_steps, feature_dim))

# 将新数据转换为 tensor 并移动到 GPU
x_new_time_series = torch.tensor(x_new_reshaped, dtype=torch.float).to(device)
y_new = torch.tensor(y_new[:batch_size_new], dtype=torch.long).to(device)  # 确保 y_new 的大小与批次大小一致

# 将新数据展平为 (batch_size_new, input_dim_new)
input_dim_new = feature_dim * time_steps
x_new_time_series = x_new_time_series.reshape(batch_size_new, input_dim_new)

# 确保打印出的形状正确
print(f"x_new_time_series shape: {x_new_time_series.shape}")  # (batch_size_new, input_dim_new)
print(f"y_new shape: {y_new.shape}")  # (batch_size_new,)

# 评估模型
evaluate_model(model, x_new_time_series, y_new)

