import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y1 = self.tcn(x)
        o = self.linear(y1[:, :, -1])
        return o

# 1. Load and Prepare Data
file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\alert.csv'
df = pd.read_csv(file_path)

# Sort by time to ensure temporal order
df = df.sort_values(by='time').reset_index(drop=True)

# Drop unnecessary columns
df.drop(columns=['Danger_Status', 'time', 'Unnamed: 0'], inplace=True)
df.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault',
                 'CompletedTrundle_Displacement_mm', 'Event_Supervisor.InspectionFault'], inplace=True)

# Keep only half of the data
df = df.iloc[:len(df)//2]

# Feature scaling
scaler = MinMaxScaler()
X = df.drop(columns=['Warning_Value'])
y = df['Warning_Value']
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))  # Scale the target as well

# 2. Reshape data for TCN with time order
timesteps = 120  # Define the number of time steps to use for each sample
X_seq = []
y_seq = []

for i in range(len(X_scaled) - timesteps):
    X_seq.append(X_scaled[i:i + timesteps])
    y_seq.append(y_scaled[i + timesteps])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Convert to PyTorch tensors
X_seq = torch.tensor(X_seq, dtype=torch.float32)
y_seq = torch.tensor(y_seq, dtype=torch.float32)

# Reshape X_seq for TCN input
X_seq = X_seq.permute(0, 2, 1)  # PyTorch expects [batch_size, input_channels, sequence_length]

# 3. Split Data based on time
split_index = int(len(X_seq) * 0.8)  # Use 80% of the data for training
X_train, X_test = X_seq[:split_index], X_seq[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

# 4. Initialize TCN Model, Loss Function, and Optimizer
input_size = X_train.shape[1]
output_size = 1
num_channels = [25, 50, 100]  # Example channel configuration
kernel_size = 3
dropout = 0.2
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCNModel(input_size, output_size, num_channels, kernel_size, dropout).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. Train the TCN Model
num_epochs = 240
batch_size = 64

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

losses = []

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0.0
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        outputs = model(batch_x).squeeze()
        loss = criterion(outputs, batch_y.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    losses.append(epoch_loss / len(X_train))

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 6. Evaluate the TCN Model
model.eval()
with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()

    mse = mean_squared_error(y_test.cpu(), y_pred)
    print(f"Mean Squared Error: {mse}")

# 7. Plot Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()

# 8. Plot Predictions vs Actual (only 1/100th of the data)
subset_size = len(y_test) // 100  # Calculate 1/100th of the data
indices = np.linspace(0, len(y_test) - 1, subset_size, dtype=int)  # Select indices to plot

plt.figure(figsize=(20, 5))
plt.plot(indices, y_test.cpu().numpy()[indices], label='Actual')
plt.plot(indices, y_pred[indices], label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Warning Value')
plt.title('Predicted vs Actual Warning Value on Test Set (1/100th of data)')
plt.legend()
plt.show()
