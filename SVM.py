import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# 1. Load and Prepare Data
file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\pca_reduced_dataset_LGAN.csv'
df = pd.read_csv(file_path)

# # Drop unnecessary columns
# df.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault'
#                    ], inplace=True)
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

new_file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\pca_reduced_dataset.csv'
df_1 = pd.read_csv(new_file_path)

# # Preprocess new data (ensure consistent processing with training data)
# df_1.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault',
#                      'time', 'time_in_seconds','time_normalized'], inplace=True)

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
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Feature Scaling
scaler = MinMaxScaler()
X = df_balanced.drop(columns=['Danger_Status'])
y = df_balanced['Danger_Status']
X_scaled = scaler.fit_transform(X)

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# 5. Model Training with Hyperparameter Tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
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
model_filename = 'best_SVM_model.joblib'
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")

# Load and use the model on new data
loaded_model = joblib.load(model_filename)
print("Model loaded successfully.")

# Process new data
new_file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\pca_reduced_dataset.csv'
new_df = pd.read_csv(new_file_path)

# # Preprocess new data (ensure consistent processing with training data)
# new_df.drop(columns=['Event_Supervisor.Combined_Fault', 'Event_Supervisor.Fault', 'Event_Supervisor.InspectionFault',
#                      'time', 'time_in_seconds', 'time_normalized'], inplace=True)
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
new_df_balanced = new_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X_new = new_df_balanced.drop(columns=['Danger_Status'])
X_new_scaled = scaler.transform(X_new)

# Predict using the loaded model
y_new_pred = loaded_model.predict(X_new_scaled)

# Save the predictions
new_df_balanced['Predicted_Danger_Status'] = y_new_pred
new_df_balanced.to_csv('dogbot429_pro_svm_test.csv', index=False)
print("Predictions saved to 'new_data_with_predictions.csv'")

# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, make_scorer, f1_score
# from sklearn.utils import resample
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.svm import SVC
#
# # 1. Load and Prepare Data
# file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\pca_reduced_dataset_LGAN.csv'
# df = pd.read_csv(file_path)
#
# # Preprocess new data
# df_1 = pd.read_csv('C:\\Users\\zesxk\\Desktop\\final project\\code\\pca_reduced_dataset.csv')
#
# # 删除 df_2 中 Danger_Status 为 0 的行
# df_2_filtered = df[df['Danger_Status'] != 0]
#
# # 合并 df_1 和过滤后的 df_2
# df = pd.concat([df_1, df_2_filtered], ignore_index=True)
#
# # 2. Handle Class Imbalance by Downsampling
# df_danger = df[df['Danger_Status'] == 1]
# df_normal = df[df['Danger_Status'] == 0]
#
# # Downsample class 0 to match class 1 size
# df_normal_downsampled = resample(df_normal,
#                                  replace=False,
#                                  n_samples=len(df_danger),
#                                  random_state=42)
#
# # Combine the downsampled class 0 and original class 1
# df_balanced = pd.concat([df_normal_downsampled, df_danger])
#
# # Shuffle the dataset
# df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # 3. Feature Scaling
# scaler = MinMaxScaler()
# X = df_balanced.drop(columns=['Danger_Status'])
# y = df_balanced['Danger_Status']
# X_scaled = scaler.fit_transform(X)
#
# # 4. Split Data
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
#
# # 5. Model Training with Hyperparameter Tuning using GridSearchCV
#
# # Define parameter grid for SVM
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': ['scale', 'auto'],
#     'kernel': ['rbf', 'linear']
# }
#
# # Define F1-Score as evaluation metric for the minority class (Danger_Status = 1)
# f1_scorer = make_scorer(f1_score, pos_label=1)
#
# # GridSearchCV with F1 score optimization for the minority class
# grid_search = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=3, scoring=f1_scorer, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)
#
# # 6. Model Evaluation
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)
#
# # Accuracy, ROC-AUC, and Classification Report
# accuracy = accuracy_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)
#
# # Output Results
# print(f"Best Hyperparameters: {grid_search.best_params_}")
# print(f"Accuracy: {accuracy}")
# print(f"ROC-AUC Score: {roc_auc}")
# print("Classification Report:\n", report)
#
# # Save the best model
# model_filename = 'best_SVM_model_f1_score.joblib'
# joblib.dump(best_model, model_filename)
# print(f"Model saved as {model_filename}")
#
# # Load and use the model on new data
# loaded_model = joblib.load(model_filename)
# print("Model loaded successfully.")
#
# # Process new data
# new_file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\pca_reduced_dataset.csv'
# new_df = pd.read_csv(new_file_path)
#
# # Handle Class Imbalance by Downsampling in new data
# new_df_danger = new_df[new_df['Danger_Status'] == 1]
# new_df_normal = new_df[new_df['Danger_Status'] == 0]
#
# # Downsample class 0 to match class 1 size
# new_df_normal_downsampled = resample(new_df_normal,
#                                  replace=False,
#                                  n_samples=len(new_df_danger) * 20,
#                                  random_state=42)
#
# # Combine the downsampled class 0 and original class 1
# new_df_balanced = pd.concat([new_df_normal_downsampled, new_df_danger])
#
# # Shuffle the dataset
# new_df_balanced = new_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
#
# X_new = new_df_balanced.drop(columns=['Danger_Status'])
# X_new_scaled = scaler.transform(X_new)
#
# # Predict using the loaded model
# y_new_pred = loaded_model.predict(X_new_scaled)
#
# # Save the predictions
# new_df_balanced['Predicted_Danger_Status'] = y_new_pred
# new_df_balanced.to_csv('dogbot429_pro_svm_f1_test.csv', index=False)
# print("Predictions saved to 'dogbot429_pro_svm_f1_test.csv'")
