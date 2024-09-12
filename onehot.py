# import pandas as pd
# import json
#
# file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot430.csv'
# df = pd.read_csv(file_path)
#
#
# # Function to parse the JSON-like 'data' field
# def parse_data_field(data):
#     try:
#         parsed = json.loads(data)
#         if isinstance(parsed, dict):
#             return parsed
#         else:
#             return {}
#     except json.JSONDecodeError:
#         return {}
#
#
# # Apply the function to the 'data' column to parse JSON data
# df['parsed_data'] = df['data'].apply(parse_data_field)
#
# # Normalize the parsed data into separate columns
# parsed_df = pd.json_normalize(df['parsed_data'])
#
# # Combine the normalized data with the original dataframe (excluding the old 'data' column)
# df_combined = pd.concat([df.drop(columns=['data', 'parsed_data']), parsed_df], axis=1)
#
# # List of specific event types to one-hot encode
# event_types_to_encode = [
#     'Supervisor.PickSuccess',
#     'Supervisor.PickFail',
#     'Supervisor.Inspected',
#     'Supervisor.FoundBerries',
#     'Supervisor.PickAttempt',
#     'Supervisor.CompletedTrundle',
#     'Supervisor.Fault',
#     'Supervisor.InspectionFault'
# ]
#
# # Filter the DataFrame to include only the specified event types
# df_filtered = df_combined[df_combined['event_id'].isin(event_types_to_encode)]
#
# # One-hot encode the event_type column with a prefix 'Event_' and ensure 0/1 format
# one_hot_encoded_event_type = pd.get_dummies(df_filtered['eevent_id'], prefix='Event').astype(int)
#
# # Combine the one-hot encoded columns back with the original DataFrame
# df_new = df.join(one_hot_encoded_event_type)
#
# # Perform one-hot encoding and add specific features for each event type as described earlier
# # Example: Supervisor.PickSuccess
# pick_success_df = df_combined[df_combined['event_id'] == 'Supervisor.PickSuccess']
# one_hot_encoded_pick_success = pd.get_dummies(pick_success_df[['Arm']], prefix='PickSuccess_Arm').astype(int)
# df_new = df_new.join(one_hot_encoded_pick_success, rsuffix='_onehot')
#
# # Example: Supervisor.PickFail
# pick_fail_df = df_combined[df_combined['event_id'] == 'Supervisor.PickFail']
# one_hot_encoded_pick_fail = pd.get_dummies(pick_fail_df[['Arm', 'Source', 'Reason']],
#                                            prefix=['PickFail_Arm', 'PickFail_Source', 'PickFail_Reason']).astype(int)
# df_new = df_new.join(one_hot_encoded_pick_fail, rsuffix='_onehot')
#
# # Example: Supervisor.Inspected
# inspected_df = df_combined[df_combined['event_id'] == 'Supervisor.Inspected']
# one_hot_encoded_inspected = pd.get_dummies(inspected_df[['Grade']], prefix='Inspected_Grade')
# inspected_df['Inspected_Mass_g'] = inspected_df['Mass_g']
# df_new = df_new.join(one_hot_encoded_inspected, rsuffix='_onehot')
# df_new = df_new.join(inspected_df[['Inspected_Mass_g']], rsuffix='_onehot')
# df_new = df_new.drop(columns=['Inspected_Grade_Grade1'])
#
# # Example: Supervisor.FoundBerries
# found_berries_df = df_combined[df_combined['event_id'] == 'Supervisor.FoundBerries']
# one_hot_encoded_found_berries = pd.get_dummies(found_berries_df[['Arm', 'Perspective']],
#                                                prefix=['FoundBerries_Arm', 'FoundBerries_Perspective']).astype(int)
# found_berries_df['FoundBerries_WorkVolumeTotalCount'] = found_berries_df['WorkVolumeTotalCount']
# found_berries_df['FoundBerries_WorkVolumeRipeCount'] = found_berries_df['WorkVolumeRipeCount']
# found_berries_df['FoundBerries_OutsideWorkVolumeTotalCount'] = found_berries_df['OutsideWorkVolumeTotalCount']
# found_berries_df['FoundBerries_OutsideWorkVolumeRipeCount'] = found_berries_df['OutsideWorkVolumeRipeCount']
# df_new = df_new.join(one_hot_encoded_found_berries, rsuffix='_onehot')
# df_new = df_new.join(found_berries_df[['FoundBerries_WorkVolumeTotalCount',
#                                        'FoundBerries_WorkVolumeRipeCount',
#                                        'FoundBerries_OutsideWorkVolumeTotalCount',
#                                        'FoundBerries_OutsideWorkVolumeRipeCount']], rsuffix='_onehot')
#
# # Example: Supervisor.PickAttempt
# pick_attempt_df = df_combined[df_combined['event_id'] == 'Supervisor.PickAttempt']
# one_hot_encoded_pick_attempt = pd.get_dummies(pick_attempt_df[['Arm']], prefix='PickAttempt_Arm').astype(int)
# pick_attempt_df['PickAttempt_RipenessProbability'] = pick_attempt_df['RipenessProbability']
# df_new = df_new.join(one_hot_encoded_pick_attempt, rsuffix='_onehot')
# df_new = df_new.join(pick_attempt_df[['PickAttempt_RipenessProbability']], rsuffix='_onehot')
#
# # Example: Supervisor.CompletedTrundle
# completed_trundle_df = df_combined[df_combined['event_id'] == 'Supervisor.CompletedTrundle']
# completed_trundle_df['CompletedTrundle_Displacement_mm'] = completed_trundle_df['Displacement_mm']
# df_new = df_new.join(completed_trundle_df[['CompletedTrundle_Displacement_mm']], rsuffix='_onehot')
#
# df_new = df_new.drop(columns=['parsed_data', 'Unnamed: 0', 'index', 'robot_name'])
# df_new['Event_Supervisor.Combined_Fault'] = df_new['Event_Supervisor.Fault'] + df_new[
#     'Event_Supervisor.InspectionFault']
#
# # Display the updated DataFrame
# print(df_new.head())
#
# df_cleaned = df_new.dropna(how='all', subset=df.columns[1:])
#
# # Fill NaN values in the specified columns using the previous value
# columns_to_fill = [
#     'Inspected_Mass_g',
#     'PickAttempt_RipenessProbability',
#     'FoundBerries_Perspective_(Global,3)',
#     'FoundBerries_WorkVolumeTotalCount',
#     'FoundBerries_WorkVolumeRipeCount',
#     'FoundBerries_OutsideWorkVolumeTotalCount',
#     'FoundBerries_OutsideWorkVolumeRipeCount'
# ]
#
# df_cleaned[columns_to_fill] = df_cleaned[columns_to_fill].fillna(method='ffill')
#
# # Convert 'time' column to datetime
# df_cleaned['time'] = pd.to_datetime(df_cleaned['time'])
#
# # Set 'time' as the index for resampling
# df_cleaned.set_index('time', inplace=True)
#
# # Resample the data to 1-minute intervals
# df_resampled = df_cleaned.resample('T').agg({
#     'Inspected_Mass_g': 'mean',
#     'PickAttempt_RipenessProbability': 'mean',
#     'FoundBerries_Perspective_(Global,3)': 'mean',
#     'FoundBerries_WorkVolumeTotalCount': 'mean',
#     'FoundBerries_WorkVolumeRipeCount': 'mean',
#     'FoundBerries_OutsideWorkVolumeTotalCount': 'mean',
#     'FoundBerries_OutsideWorkVolumeRipeCount': 'mean',
#     **{col: 'sum' for col in df_cleaned.columns if col not in [
#         'Inspected_Mass_g',
#         'PickAttempt_RipenessProbability',
#         'FoundBerries_Perspective_(Global,3)',
#         'FoundBerries_WorkVolumeTotalCount',
#         'FoundBerries_WorkVolumeRipeCount',
#         'FoundBerries_OutsideWorkVolumeTotalCount',
#         'FoundBerries_OutsideWorkVolumeRipeCount'
#     ]}
# })
#
# # Reset the index to bring 'time' back as a column
# df_resampled.reset_index(inplace=True)
# df_final = df_resampled.dropna()
#
#
# # Create a function to fill in the zeros with the most recent non-zero value
# def fill_zeros_with_last_value(series):
#     last_value = None
#     for i in range(len(series)):
#         if series[i] != 0 and not pd.isna(series[i]):
#             last_value = series[i]
#         elif series[i] == 0:
#             series[i] = last_value
#     return series
#
#
# # Apply the function to the 'CompletedTrundle_Displacement_mm' column
# df_final['CompletedTrundle_Displacement_mm'] = fill_zeros_with_last_value(
#     df_final['CompletedTrundle_Displacement_mm'].values)
# df_final = df_final.dropna()
#
# # 初始化 'Danger_Status' 列，默认值为 0（正常状态）
# df_final['Danger_Status'] = 0
#
# # 获取不为0的 'Event_Supervisor.Combined_Fault' 的索引
# fault_indices = df_final.index[df_final['Event_Supervisor.Combined_Fault'] != 0]
#
# # 遍历这些索引，并将前一个时间点的 'Danger_Status' 标记为 1（危险状态）
# for fault_index in fault_indices:
#     # 找到当前索引的前一个位置
#     prev_index = df_final.index.get_loc(fault_index) - 1
#
#     # 检查是否超出范围
#     if prev_index >= 0:
#         df_final.iloc[prev_index, df_final.columns.get_loc('Danger_Status')] = 1
#
# # 检查结果
#
# df_final.to_csv('C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot430_pycharm.csv')

import pandas as pd
import json
file_path = 'C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot445_2024.csv'
# 将 time 列转换为 datetime 对象
df = pd.read_csv(file_path)
df['time'] = pd.to_datetime(df['time'])



# # 提取特定年份的数据，例如 2022 年
# year_to_filter = 2023
# df_filtered = df[df['time'].dt.year == year_to_filter]

# # 按照时间顺序排序
# df = df_filtered.sort_values(by='time')

# Function to parse the JSON-like 'data' field
def parse_data_field(data):
    try:
        parsed = json.loads(data)
        if isinstance(parsed, dict):
            return parsed
        else:
            return {}
    except json.JSONDecodeError:
        return {}


# Apply the function to the 'data' column to parse JSON data
df['parsed_data'] = df['data'].apply(parse_data_field)

# Normalize the parsed data into separate columns
parsed_df = pd.json_normalize(df['parsed_data'])

# Combine the normalized data with the original dataframe (excluding the old 'data' column)
df_combined = pd.concat([df.drop(columns=['data', 'parsed_data']), parsed_df], axis=1)

# List of specific event types to one-hot encode
event_types_to_encode = [
    'Supervisor.PickSuccess',
    'Supervisor.PickFail',
    'Supervisor.Inspected',
    'Supervisor.FoundBerries',
    'Supervisor.PickAttempt',
    'Supervisor.CompletedTrundle',
    'Supervisor.Fault',
    'Supervisor.InspectionFault'
]

# Filter the DataFrame to include only the specified event types
df_filtered = df_combined[df_combined['event_id'].isin(event_types_to_encode)]

# One-hot encode the event_type column with a prefix 'Event_' and ensure 0/1 format
one_hot_encoded_event_type = pd.get_dummies(df_filtered['event_id'], prefix='Event').astype(int)
df_new = df.join(one_hot_encoded_event_type)

# Perform one-hot encoding and add specific features for each event type as described earlier
# Example: Supervisor.PickSuccess
pick_success_df = df_combined[df_combined['event_id'] == 'Supervisor.PickSuccess']
one_hot_encoded_pick_success = pd.get_dummies(pick_success_df[['Arm']], prefix='PickSuccess_Arm').astype(int)
df_new = df_new.join(one_hot_encoded_pick_success, rsuffix='_onehot')

# Example: Supervisor.PickFail
pick_fail_df = df_combined[df_combined['event_id'] == 'Supervisor.PickFail']
one_hot_encoded_pick_fail = pd.get_dummies(pick_fail_df[['Arm', 'Source', 'Reason']],
                                           prefix=['PickFail_Arm', 'PickFail_Source', 'PickFail_Reason']).astype(int)
df_new = df_new.join(one_hot_encoded_pick_fail, rsuffix='_onehot')

# Example: Supervisor.Inspected
inspected_df = df_combined[df_combined['event_id'] == 'Supervisor.Inspected']
# one_hot_encoded_inspected = pd.get_dummies(inspected_df[['Grade']], prefix='Inspected_Grade')
inspected_df['Inspected_Mass_g'] = inspected_df['Mass']
# df_new = df_new.join(one_hot_encoded_inspected, rsuffix='_onehot')
df_new = df_new.join(inspected_df[['Inspected_Mass_g']], rsuffix='_onehot')
# df_new = df_new.drop(columns=['Inspected_Grade_Grade1'])

df_new['Inspected_Mass_g'] = (
    df_new['Inspected_Mass_g']
    .str.replace('g', '', regex=False)  # Remove the 'mm' suffix
    .replace('', pd.NA)  # Replace empty strings with NaN
)
df_new['Inspected_Mass_g'] = pd.to_numeric(df_new['Inspected_Mass_g'], errors='coerce')
print(df_new['Inspected_Mass_g'])

# Example: Supervisor.FoundBerries
found_berries_df = df_combined[df_combined['event_id'] == 'Supervisor.FoundBerries']
one_hot_encoded_found_berries = pd.get_dummies(found_berries_df[['Arm', 'Perspective']],
                                               prefix=['FoundBerries_Arm', 'FoundBerries_Perspective']).astype(int)
found_berries_df['FoundBerries_WorkVolumeTotalCount'] = found_berries_df['WorkVolumeTotalCount']
found_berries_df['FoundBerries_WorkVolumeRipeCount'] = found_berries_df['WorkVolumeRipeCount']
found_berries_df['FoundBerries_OutsideWorkVolumeTotalCount'] = found_berries_df['OutsideWorkVolumeTotalCount']
found_berries_df['FoundBerries_OutsideWorkVolumeRipeCount'] = found_berries_df['OutsideWorkVolumeRipeCount']
df_new = df_new.join(one_hot_encoded_found_berries, rsuffix='_onehot')
df_new = df_new.join(found_berries_df[['FoundBerries_WorkVolumeTotalCount',
                                       'FoundBerries_WorkVolumeRipeCount',
                                       'FoundBerries_OutsideWorkVolumeTotalCount',
                                       'FoundBerries_OutsideWorkVolumeRipeCount']], rsuffix='_onehot')

# Example: Supervisor.PickAttempt
pick_attempt_df = df_combined[df_combined['event_id'] == 'Supervisor.PickAttempt']
one_hot_encoded_pick_attempt = pd.get_dummies(pick_attempt_df[['Arm']], prefix='PickAttempt_Arm').astype(int)
pick_attempt_df['PickAttempt_RipenessProbability'] = pick_attempt_df['RipenessProbability']
df_new = df_new.join(one_hot_encoded_pick_attempt, rsuffix='_onehot')
df_new = df_new.join(pick_attempt_df[['PickAttempt_RipenessProbability']], rsuffix='_onehot')

# Example: Supervisor.CompletedTrundle
completed_trundle_df = df_combined[df_combined['event_id'] == 'Supervisor.CompletedTrundle']
completed_trundle_df['CompletedTrundle_Displacement_mm'] = completed_trundle_df['Displacement_mm']
df_new = df_new.join(completed_trundle_df[['CompletedTrundle_Displacement_mm']], rsuffix='_onehot')

df_new['CompletedTrundle_Displacement_mm'] = (
    df_new['CompletedTrundle_Displacement_mm']
    .str.replace('mm', '', regex=False)  # Remove the 'mm' suffix
    .replace('', pd.NA)  # Replace empty strings with NaN
)

# Example: Supervisor.Levelled
Levelled_df = df_combined[df_combined['event_id'] == 'Supervisor.Levelled']
Levelled_df['Levelled_RollDegrees'] = Levelled_df['RollDegrees']
# df_new = df_new.join(one_hot_encoded_inspected, rsuffix='_onehot')
df_new = df_new.join(Levelled_df[['Levelled_RollDegrees']], rsuffix='_onehot')

# Example: YieldForecastingEvent.SectorBerriesObservation
pick_fail_df = df_combined[df_combined['event_id'] == 'YieldForecastingEvent.SectorBerriesObservation']
one_hot_encoded_pick_fail = pd.get_dummies(pick_fail_df[['Mode', 'Arm', 'Side']],
                                           prefix=['SectorBerriesObservation_Mode', 'SectorBerriesObservation_Arm', 'SectorBerriesObservation_Side']).astype(int)
df_new = df_new.join(one_hot_encoded_pick_fail, rsuffix='_onehot')

df_new = df_new.drop(columns=['parsed_data', 'robot_name'])
df_new['Event_Supervisor.Combined_Fault'] = df_new['Event_Supervisor.Fault'] + df_new[
    'Event_Supervisor.InspectionFault']

# Display the updated DataFrame


df_new = df_new.drop(columns=['id', 'event_id', 'data'])
df_cleaned = df_new.dropna(how='all', subset=df_new.columns[1:])



# Fill NaN values in the specified columns using the previous value
columns_to_fill = [
    'Inspected_Mass_g',
    'PickAttempt_RipenessProbability',
    'FoundBerries_Perspective_(Global,3)',
    'FoundBerries_WorkVolumeTotalCount',
    'FoundBerries_WorkVolumeRipeCount',
    'FoundBerries_OutsideWorkVolumeTotalCount',
    'FoundBerries_OutsideWorkVolumeRipeCount',
    'Levelled_RollDegrees'
]

df_cleaned[columns_to_fill] = df_cleaned[columns_to_fill].fillna(method='ffill')

# Set 'time' as the index for resampling
df_cleaned.set_index('time', inplace=True)
print(df_cleaned.dtypes)


# Resample the data to 1-minute intervals
df_resampled = df_cleaned.resample('T').agg({
    'Inspected_Mass_g': 'mean',
    'PickAttempt_RipenessProbability': 'mean',
    'FoundBerries_Perspective_(Global,3)': 'mean',
    'FoundBerries_WorkVolumeTotalCount': 'mean',
    'FoundBerries_WorkVolumeRipeCount': 'mean',
    'FoundBerries_OutsideWorkVolumeTotalCount': 'mean',
    'FoundBerries_OutsideWorkVolumeRipeCount': 'mean',
    'Levelled_RollDegrees': 'mean',
    **{col: 'sum' for col in df_cleaned.columns if col not in [
        'Inspected_Mass_g',
        'PickAttempt_RipenessProbability',
        'FoundBerries_Perspective_(Global,3)',
        'FoundBerries_WorkVolumeTotalCount',
        'FoundBerries_WorkVolumeRipeCount',
        'FoundBerries_OutsideWorkVolumeTotalCount',
        'FoundBerries_OutsideWorkVolumeRipeCount',
        'Levelled_RollDegrees'
    ]}
})

# Reset the index to bring 'time' back as a column
df_resampled.reset_index(inplace=True)
df_final = df_resampled.dropna()


# Create a function to fill in the zeros with the most recent non-zero value
def fill_zeros_with_last_value(series):
    last_value = None
    for i in range(len(series)):
        if series[i] != 0 and not pd.isna(series[i]):
            last_value = series[i]
        elif series[i] == 0:
            series[i] = last_value
    return series


# Apply the function to the 'CompletedTrundle_Displacement_mm' column
df_final['CompletedTrundle_Displacement_mm'] = fill_zeros_with_last_value(
    df_final['CompletedTrundle_Displacement_mm'].values)
df_final = df_final.dropna()

# 初始化 'Danger_Status' 列，默认值为 0（正常状态）
df_final['Danger_Status'] = 0

# 获取不为0的 'Event_Supervisor.Combined_Fault' 的索引
fault_indices = df_final.index[df_final['Event_Supervisor.Combined_Fault'] != 0]

# 遍历这些索引，并将前一个时间点的 'Danger_Status' 标记为 1（危险状态）
for fault_index in fault_indices:
    # 找到当前索引的前一个位置
    prev_index = df_final.index.get_loc(fault_index) - 1
    # 三次样条？？

    # 检查是否超出范围
    if prev_index >= 0:
        df_final.iloc[prev_index, df_final.columns.get_loc('Danger_Status')] = 1

df_final['Inspected_Mass_g'] = df_final['Inspected_Mass_g'].replace(to_replace=0, method='ffill')
df_final = df_final[df_final['Inspected_Mass_g'] != 0]
df_final = df_final.drop(columns=['Event_Supervisor.CompletedTrundle','Event_Supervisor.Inspected','SectorBerriesObservation_Mode_Pick', 'CompletedTrundle_Displacement_mm'])
# df_final = df_final.drop(columns=[df_final.columns[0]])

# # Keep rows where 'CompletedTrundle_Displacement_mm' is one of the allowed values
# allowed_values = [250, -250, 1750, -1750]
# df_final = df_final[df_final['CompletedTrundle_Displacement_mm'].isin(allowed_values)]

# # Create a new column for the warning score
# df_final['Warning_Score'] = 0
#
# # Find indices where 'Danger_Status' is 1
# fault_indices = df_final[df_final['Danger_Status'] == 1].index

# # Assign warning scores based on proximity to the next fault event
# for i in range(len(fault_indices)):
#     if i == 0:
#         prev_idx = 0
#     else:
#         prev_idx = fault_indices[i - 1] + 1
#
#     current_idx = fault_indices[i]
#
#     time_to_fault = (df_final.loc[prev_idx:current_idx, 'time'] - df_final.loc[current_idx, 'time']).dt.total_seconds().abs()
#     max_time = time_to_fault.max()  # Find the maximum time difference in this segment
#
#     # Calculate warning scores: inversely proportional to the time to fault
#     df_final.loc[prev_idx:current_idx, 'Warning_Score'] = (1 - time_to_fault / max_time) *100
#
# df_final = df_final.fillna(0)

# 将时间转换为一天中的秒数
df_final['time_in_seconds'] = df_final['time'].dt.hour * 3600 + df_final['time'].dt.minute * 60 + df_final['time'].dt.second

# 一天的总秒数
seconds_in_a_day = 24 * 60 * 60

# 将时间转换为一天中的比例（归一化为0到1之间的值）
df_final['time_normalized'] = df_final['time_in_seconds'] / seconds_in_a_day

# 查看数据框
print(df_final[['time', 'time_in_seconds', 'time_normalized']].head())

print(df_final.head())

df_final.to_csv('C:\\Users\\zesxk\\Desktop\\final project\\code\\dogbot445_pro.csv', index=False)