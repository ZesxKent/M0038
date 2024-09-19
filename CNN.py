import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import joblib
import torch.nn.functional as F
# 1. Load and Prepare Data
# 1. Load and Prepare Data
# file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot429_CGAN_cuda.csv'
# df = pd.read_csv(file_path)
#
# # Drop unnecessary columns
# df.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault'
#                  ,'time_normalized'], inplace=True)
#
# float_columns = [
#     'Inspected_Mass_g', 'PickAttempt_RipenessProbability',
#     'FoundBerries_Perspective_(Global,3)', 'FoundBerries_WorkVolumeTotalCount',
#     'FoundBerries_WorkVolumeRipeCount', 'FoundBerries_OutsideWorkVolumeTotalCount',
#     'FoundBerries_OutsideWorkVolumeRipeCount', 'Levelled_RollDegrees'
# ]
# df.fillna(df.median(), inplace=True)
# # 找出所有需要转换为整数的列（即不在上述float_columns中的列）
# integer_columns = [col for col in df.columns if col not in float_columns]
#
# # 对这些列进行四舍五入并转换为整数
# for col in integer_columns:
#     df[col] = np.round(df[col]).astype(int)
#
# # 如果有必要，可以对特定的列使用clip方法以确保合理范围
#
# for col in integer_columns:
#     df[col] = df[col].clip(lower=0)  # 假设下限是0，根据实际情况调整

new_file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot429_pro.csv'
df_1 = pd.read_csv(new_file_path)

# Preprocess new data (ensure consistent processing with training data)
df_1.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault',
                   'time', 'time_in_seconds', 'time_normalized'], inplace=True)
#
# # 删除 df_2 中 Danger_Status 为 0 的行
# df_2_filtered = df[df['Danger_Status'] != 0]
#
# # 合并 df_1 和过滤后的 df_2
# df = pd.concat([df_1, df_2_filtered], ignore_index=True)
df = df_1
# 查看合并后的数据
print(df)

# 2. Handle Class Imbalance by Downsampling
df_danger = df[df['Danger_Status'] == 1]
df_normal = df[df['Danger_Status'] == 0]

# Downsample class 0 to match class 1 size
df_normal_downsampled = resample(df_normal, replace=False, n_samples=len(df_danger), random_state=42)
df_balanced = pd.concat([df_normal_downsampled, df_danger])

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"df_balanced 样本总数: {len(df_balanced)}")
print(f"df_danger (class 1) 样本数量: {len(df_danger)}")
print(f"df_normal (class 0) 样本数量: {len(df_normal)}")

# 3. Feature Scaling
scaler = MinMaxScaler()
X = df_balanced.drop(columns=['Danger_Status'])
y = df_balanced['Danger_Status']
print(X)
# Reshape data for CNN: Assuming 1D input for each sample
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.3, random_state=42, stratify=y)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 5. Define CNN Model
class CNN_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # Calculate the flattened input size for the first fully connected layer
        # input_size needs to be adjusted based on the output size after conv + pooling
        conv_output_size = input_size // 4  # Adjusting for pooling layer

        self.fc1 = nn.Linear(32 * conv_output_size, 64)  # Adjusted to match the flattened conv output
        self.fc2 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


input_size = X_train.shape[2]
output_size = 2  # Binary classification (Danger_Status: 0, 1)

model = CNN_Model(input_size, output_size)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 6. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)


# 7. Training Function
def train(model, criterion, optimizer, train_loader, epochs=50):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    model = model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")


# Train the model
train(model, criterion, optimizer, train_loader)


# 8. Evaluation Function
def evaluate(model, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model = model.to(device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"ROC-AUC Score: {roc_auc}")
    print("Classification Report:\n", report)


# Evaluate the model
evaluate(model, test_loader)

# Save the trained model
torch.save(model.state_dict(), 'cnn_model.pth')
print("Model saved as cnn_model.pth")


# Load and use the model on new data
def predict_on_new_data(model, X_new):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    with torch.no_grad():
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)
        outputs = model(X_new_tensor)
        _, predictions = torch.max(outputs, 1)
    return predictions.cpu().numpy()


# Process new data
new_file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot429_pro.csv'
new_df = pd.read_csv(new_file_path)

# Preprocess new data (ensure consistent processing with training data)
new_df.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault',
                     'time', 'time_in_seconds', 'time_normalized'], inplace=True)

# Handle class imbalance by downsampling
new_df_danger = new_df[new_df['Danger_Status'] == 1]
new_df_normal = new_df[new_df['Danger_Status'] == 0]
new_df_normal_downsampled = resample(new_df_normal, replace=False, n_samples=len(new_df_danger) * 2, random_state=42)
new_df_balanced = pd.concat([new_df_normal_downsampled, new_df_danger])

# Shuffle the dataset
new_df_balanced = new_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare the new data
X_new = new_df_balanced.drop(columns=['Danger_Status'])
X_new_scaled = scaler.transform(X_new)
X_new_reshaped = X_new_scaled.reshape(X_new_scaled.shape[0], 1, X_new_scaled.shape[1])

# Load the trained model
loaded_model = CNN_Model(input_size, output_size)
loaded_model.load_state_dict(torch.load('cnn_model.pth'))

# Predict on new data
y_new_pred = predict_on_new_data(loaded_model, X_new_reshaped)

# Save the predictions
new_df_balanced['Predicted_Danger_Status'] = y_new_pred
new_df_balanced.to_csv('dogbot429_pro_CNN_test.csv', index=False)
print("Predictions saved")
