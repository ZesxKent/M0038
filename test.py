import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载保存的预测结果
predicted_df = pd.read_csv('C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot429_pro_RF_test.csv')

# 2. 查看数据集的前几行，检查预测结果
print(predicted_df.head())

# 3. 查看预测结果的分布
prediction_distribution = predicted_df['Predicted_Danger_Status'].value_counts()
print("Prediction Distribution:\n", prediction_distribution)

# 4. 与真实标签进行对比（如果新数据中包含真实标签）
if 'Danger_Status' in predicted_df.columns:
    y_true = predicted_df['Danger_Status']
    y_pred = predicted_df['Predicted_Danger_Status']

    # 计算并输出各种性能指标
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"ROC-AUC Score: {roc_auc}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
else:
    print("No true Danger_Status labels found in the dataset.")

# 5. 可视化混淆矩阵
if 'Danger_Status' in predicted_df.columns:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Normal', 'Predicted Danger'],
                yticklabels=['Actual Normal', 'Actual Danger'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# 6. 分析错分样本
if 'Danger_Status' in predicted_df.columns:
    # 识别错分样本
    incorrect_predictions = predicted_df[predicted_df['Danger_Status'] != predicted_df['Predicted_Danger_Status']]

    # 查看错分样本的特征分布
    print("Incorrect Predictions:\n", incorrect_predictions.head())

# # 7. 按时间段分析模型表现（假设数据集中包含时间列）
# if 'time' in predicted_df.columns:
#     predicted_df['time'] = pd.to_datetime(predicted_df['time'])
#     predicted_df['hour'] = predicted_df['time'].dt.hour
#
#     # 计算每小时的模型表现（准确率）
#     hourly_performance = predicted_df.groupby('hour').apply(
#         lambda x: accuracy_score(x['Danger_Status'], x['Predicted_Danger_Status'])
#     )
#     print("Hourly Performance:\n", hourly_performance)
