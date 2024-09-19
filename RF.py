import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
import joblib

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

new_file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot429_pro.csv'
df_1 = pd.read_csv(new_file_path)

# Preprocess new data (ensure consistent processing with training data)
df_1.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault',
                     'time', 'time_in_seconds','time_normalized'], inplace=True)

# 删除 df_2 中 Danger_Status 为 0 的行
df_2_filtered = df[df['Danger_Status'] != 0]

# 合并 df_1 和过滤后的 df_2
df = pd.concat([df_1, df_2_filtered], ignore_index=True)

# 查看合并后的数据
print(df)

# 2. Handle Class Imbalance by Downsampling
df_danger = df[df['Danger_Status'] == 1]
df_normal = df[df['Danger_Status'] == 0]

# Downsample class 0 to match class 1 size
df_normal_downsampled = resample(df_normal,
                                 replace=False,
                                 n_samples=len(df_danger),
                                 random_state=42)

# Combine the downsampled class 0 and original class 1
df_balanced = pd.concat([df_normal_downsampled, df_danger])

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=4).reset_index(drop=True)

# 3. Feature Scaling
scaler = MinMaxScaler()
X = df_balanced.drop(columns=['Danger_Status'])
y = df_balanced['Danger_Status']
print(X)
print(y)
X_scaled = scaler.fit_transform(X)

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# 5. Model Training with Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt']
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
                           param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 6. Model Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Accuracy and ROC-AUC Score
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 7. Output Results
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy}")
print(f"ROC-AUC Score: {roc_auc}")
print("Classification Report:\n", report)

# 保存最佳模型
model_filename = 'best_random_forest_model.joblib'
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")

# Load and use the model on new data
loaded_model = joblib.load(model_filename)
print("Model loaded successfully.")

# Process new data
new_file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot429_pro.csv'
new_df = pd.read_csv(new_file_path)

# Preprocess new data (ensure consistent processing with training data)
new_df.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault',
                     'time', 'time_in_seconds','time_normalized'], inplace=True)
# 2. Handle Class Imbalance by Downsampling
new_df_danger = new_df[new_df['Danger_Status'] == 1]
new_df_normal = new_df[new_df['Danger_Status'] == 0]

# Downsample class 0 to match class 1 size
new_df_normal_downsampled = resample(new_df_normal,
                                 replace=False,
                                 n_samples=len(new_df_danger) * 20,
                                 random_state=42)

# Combine the downsampled class 0 and original class 1
new_df_balanced = pd.concat([new_df_normal_downsampled, new_df_danger])

# Shuffle the dataset
new_df_balanced = new_df_balanced.sample(frac=1, random_state=4).reset_index(drop=True)

X_new = new_df_balanced.drop(columns=['Danger_Status'])
X_new_scaled = scaler.transform(X_new)

# Predict using the loaded model
y_new_pred = loaded_model.predict(X_new_scaled)

# Save the predictions
new_df_balanced['Predicted_Danger_Status'] = y_new_pred
new_df_balanced.to_csv('dogbot429_pro_RF_test.csv', index=False)
print("Predictions saved")