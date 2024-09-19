import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 1. 加载数据
df = pd.read_csv('C:\\Users\\zesxk\\Desktop\\final project\\code\\pca_reduced_dataset.csv')
# df.drop(columns=['time', 'time_in_seconds'], inplace=True)

# 4. 提取少数类（Danger_Status = 1）和多数类（Danger_Status = 0）样本
minority_class = df[df['Danger_Status'] == 1]
majority_class = df[df['Danger_Status'] == 0]

# 5. 设置目标数量，1:20 的比例
minority_target_count = 10000 # 少数类目标数量
majority_target_count = 10000  # 多数类目标数量 (1:20 比例)

# 6. 如果少数类样本少于目标数量，使用LGAN生成更多样本
if len(minority_class) < minority_target_count:
    # 提取并标准化少数类数据
    minority_data = minority_class.drop(columns=['Danger_Status']).values
    mean = minority_data.mean(axis=0)
    std = minority_data.std(axis=0)
    minority_data = (minority_data - mean) / std
    minority_tensor = torch.tensor(minority_data, dtype=torch.float32).to(device)

    # 数据加载器
    batch_size = 64
    dataset = TensorDataset(minority_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义生成器模型 (基于LSTM)
    class LSTMGenerator(nn.Module):
        def __init__(self, latent_dim, output_dim, hidden_dim, num_layers):
            super(LSTMGenerator, self).__init__()
            self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, z):
            h0 = torch.zeros(num_layers, z.size(0), hidden_dim).to(z.device)
            c0 = torch.zeros(num_layers, z.size(0), hidden_dim).to(z.device)
            out, _ = self.lstm(z, (h0, c0))
            out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
            return out

    # 定义判别器模型
    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.model(x)

    # 初始化LGAN模型
    latent_dim = 100
    hidden_dim = 256
    num_layers = 2
    input_dim = minority_data.shape[1]
    generator = LSTMGenerator(latent_dim, input_dim, hidden_dim, num_layers).to(device)
    discriminator = Discriminator(input_dim).to(device)

    # 损失函数和优化器
    loss_function = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    # 训练LGAN模型
    n_epochs = 3000  # 训练次数可根据需求调整
    for epoch in range(n_epochs):
        for i, data in enumerate(data_loader):
            real_samples = data[0].to(device)
            batch_size = real_samples.size(0)

            # 训练判别器
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, 1, latent_dim).to(device)  # 为LSTM调整输入形状为 (batch_size, 时间步长, 特征)
            fake_samples = generator(z)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            real_loss = loss_function(discriminator(real_samples), real_labels)
            fake_loss = loss_function(discriminator(fake_samples.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, 1, latent_dim).to(device)  # 同样为生成器输入形状调整
            fake_samples = generator(z)
            g_loss = loss_function(discriminator(fake_samples), real_labels)
            g_loss.backward()
            optimizer_G.step()

        if epoch % 1000 == 0:
            print(f'Epoch {epoch} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}')

    # 生成额外的少数类样本
    additional_samples_needed = minority_target_count - len(minority_class)
    z = torch.randn(additional_samples_needed, 1, latent_dim).to(device)
    new_samples = generator(z).detach().cpu().numpy()

    # 将生成的新样本还原到原始尺度
    new_samples = new_samples * std + mean
    new_samples_df = pd.DataFrame(new_samples, columns=minority_class.columns[:-1])
    new_samples_df['Danger_Status'] = 1  # 添加 Danger_Status 列，值为 1

    # 合并原始少数类样本和新生成的样本
    minority_class = pd.concat([minority_class, new_samples_df])

# 7. 随机抽样少数类样本到目标数量（2500条）
minority_class = minority_class.sample(n=minority_target_count, random_state=42)

# 8. 随机抽样多数类样本到目标数量（50000条）
if len(majority_class) >= majority_target_count:
    # 如果多数类样本足够，则无需替换
    majority_class = majority_class.sample(n=majority_target_count, random_state=42)
else:
    # 如果多数类样本不足，使用替换方式采样
    majority_class = majority_class.sample(n=majority_target_count, replace=True, random_state=42)

# 9. 合并少数类和多数类样本形成新的平衡数据集
balanced_df = pd.concat([minority_class, majority_class])

# 10. 保存为新的CSV文件
balanced_df.to_csv('C:\\Users\\zesxk\\Desktop\\final project\\code\\pca_reduced_dataset_LGAN.csv', index=False)

# 11. 查看生成后数据的平衡情况
print(balanced_df['Danger_Status'].value_counts())
