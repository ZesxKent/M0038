import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 1. 加载数据
df = pd.read_csv('C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot429_pro_5min.csv')
df.drop(columns=['time', 'time_in_seconds'], inplace=True)

# 4. 提取少数类（Danger_Status = 1）和多数类（Danger_Status = 0）样本
minority_class = df[df['Danger_Status'] == 1]
majority_class = df[df['Danger_Status'] == 0]

# 5. 设置目标数量，1:20 的比例
minority_target_count = 10000  # 少数类目标数量
majority_target_count = 10000  # 多数类目标数量 (1:20 比例)

# 6. 如果少数类样本少于目标数量，使用CGAN生成更多样本
if len(minority_class) < minority_target_count:
    # 提取并标准化少数类数据
    minority_data = minority_class.drop(columns=['Danger_Status']).values
    mean = minority_data.mean(axis=0)
    std = minority_data.std(axis=0)
    minority_data = (minority_data - mean) / std
    minority_tensor = torch.tensor(minority_data, dtype=torch.float32).to(device)

    # 条件标签
    minority_labels = torch.ones(len(minority_class), 1).to(device)

    # 数据加载器
    batch_size = 64
    dataset = TensorDataset(minority_tensor, minority_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # 定义生成器模型
    class Generator(nn.Module):
        def __init__(self, latent_dim, output_dim, condition_dim):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_dim + condition_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim),
            )

        def forward(self, z, condition):
            input = torch.cat([z, condition], dim=1)
            return self.model(input)


    # 定义判别器模型
    class Discriminator(nn.Module):
        def __init__(self, input_dim, condition_dim):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim + condition_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        def forward(self, x, condition):
            input = torch.cat([x, condition], dim=1)
            return self.model(input)


    # 初始化CGAN模型
    latent_dim = 100
    input_dim = minority_data.shape[1]
    condition_dim = 1  # Danger_Status 有两种状态（0 和 1）
    generator = Generator(latent_dim, input_dim, condition_dim).to(device)
    discriminator = Discriminator(input_dim, condition_dim).to(device)

    # 损失函数和优化器
    loss_function = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    # 训练CGAN模型
    n_epochs = 5000  # 训练次数可根据需求调整
    for epoch in range(n_epochs):
        for i, (real_samples, labels) in enumerate(data_loader):
            batch_size = real_samples.size(0)

            # 训练判别器
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_samples = generator(z, labels)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            real_loss = loss_function(discriminator(real_samples, labels), real_labels)
            fake_loss = loss_function(discriminator(fake_samples.detach(), labels), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_samples = generator(z, labels)
            g_loss = loss_function(discriminator(fake_samples, labels), real_labels)
            g_loss.backward()
            optimizer_G.step()

        if epoch % 1000 == 0:
            print(f'Epoch {epoch} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}')

    # 生成额外的少数类样本
    additional_samples_needed = minority_target_count - len(minority_class)
    z = torch.randn(additional_samples_needed, latent_dim).to(device)
    condition = torch.ones(additional_samples_needed, 1).to(device)
    new_samples = generator(z, condition).detach().cpu().numpy()

    # 将生成的新样本还原到原始尺度
    new_samples = new_samples * std + mean
    new_samples_df = pd.DataFrame(new_samples, columns=minority_class.columns[:-1])
    new_samples_df['Danger_Status'] = 1  # 添加 Danger_Status 列，值为 1

    # 合并原始少数类样本和新生成的样本
    minority_class = pd.concat([minority_class, new_samples_df])

# 7. 随机抽样少数类样本到目标数量（10000条）
minority_class = minority_class.sample(n=minority_target_count, random_state=42)

# 8. 随机抽样多数类样本到目标数量（10000条）
if len(majority_class) >= majority_target_count:
    majority_class = majority_class.sample(n=majority_target_count, random_state=42)
else:
    majority_class = majority_class.sample(n=majority_target_count, replace=True, random_state=42)

# 9. 合并少数类和多数类样本形成新的平衡数据集
balanced_df = pd.concat([minority_class, majority_class])

# 10. 保存为新的CSV文件
balanced_df.to_csv('C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot429_LGAN_5min.csv', index=False)

# 11. 查看生成后数据的平衡情况
print(balanced_df['Danger_Status'].value_counts())
