import pandas as pd
import matplotlib.pyplot as plt
import os
import string
import re
import pickle

from scipy.stats import zscore
from scipy.stats import spearmanr
from sklearn.preprocessing import  OneHotEncoder

#直接讀取
path = '2022-06-09data/'
path1  = '2022-06-09data/'

# directory = '2022-06-09data/test3_2.xlsx'
directory1 = '2022-06-09data/finaltest1.xlsx'
directory2 = '2022-06-09data/Newtest1.xlsx'
directory3 = '2022-06-09data/finaltest.xlsx'
directory4 = '2024-01-15/Newtest3goal.xlsx'
# 'FinalcountryMerged_kickstarter_data.xlsx' FinalMerged_kickstarter_data.xlsx

# df = pd.read_excel(directory)
df1 = pd.read_excel(directory1)   #預測資料
df2 = pd.read_excel(directory2)  #6萬多筆資料
df3 = pd.read_excel(directory3)
df4 = pd.read_excel(directory4)
print("Check file is okokokokokokokokokokok")

# 檢查state 沒有亂碼
# unique_states = df['state'].unique()
# non_standard_states = [state for state in unique_states if state not in ['successful', 'failed']]
# print(f"States other than 'successful' or 'failed': {non_standard_states}")

#轉換時間 將不是數字格式的值刪除
def process_file(file_path, start_date, end_date, encoding='ISO-8859-1'):
    """读取并处理单个文件，返回清理后的数据。"""
    df = pd.read_csv(file_path, encoding=encoding)
    df.dropna(subset=['deadline', 'launched_at'], inplace=True) #確定沒有缺失值

    # 转换为 datetime 并过滤日期  把不合理排除
    df['deadline'] = pd.to_datetime(pd.to_numeric(df['deadline'], errors='coerce'), unit='s') #数值是以秒为单位的 Unix 时间戳， pd.to_datetime 轉化時間
    df['launched_at'] = pd.to_datetime(pd.to_numeric(df['launched_at'], errors='coerce'), unit='s')
    return df[(df['deadline'] <= end_date) & (df['launched_at'] >= start_date) & (df['deadline']>=df['launched_at'])]
# #
# # # #餵入兩筆不同名稱格式的值，並將其合併
directory = '2022-06-09data/'
df = pd.read_excel(directory)

start_date = pd.to_datetime('2020-01-01')
end_date = pd.to_datetime('2023-06-09')
file_paths = [os.path.join(directory, f'Kickstarter{i:03}.csv') for i in range(1, 63)] + [
    os.path.join(directory, 'Kickstarter.csv')]
# #
all_data = []
for file_path in file_paths:
    if os.path.exists(file_path):
        df_cleaned = process_file(file_path, start_date)
        all_data.append(df_cleaned)
    else:
        print(f"{file_path} not found.")

merged_df = pd.concat(all_data, ignore_index=True)
merged_df.drop_duplicates(inplace=True)    #移除重複的值
# # #
# # # # 选择指定欄位
selected_columns = [
    'launched_at', 'deadline', 'pledged',
    'usd_pledged', 'goal', 'category','source_url',
    'backers_count', 'state','blurb','country_displayable_name'
]
final_df = merged_df[selected_columns]
# # #
# # # # 保存到新的 Excel 文件
# output_file_path = os.path.join(directory, 'FinalMerged_kickstarter_data.xlsx')
# final_df.to_excel(output_file_path, index=False)

# 這裡的df 改用上面新的excel名稱
#新增 duration 欄位
df['launched_at'] = pd.to_datetime(df['launched_at'], format='%Y/%m/%d %I:%M:%S %p')
df['deadline'] = pd.to_datetime(df['deadline'], format='%Y/%m/%d %I:%M:%S %p')
df['duration'] = df['deadline'] - df['launched_at']
#
# #更新category 取 source_url內的值代替 並且將 film後面的值刪除
df['category'] = df['source_url'].str.extract(r'/categories/([^/]+)')
df['category'] = df['category'].str.replace(r'^film.*', 'film', regex=True)
#
# # df_filtered = df[df['state'] != 'live']                 #  .....檢查用
# # cancel_test = df[df['state'] == 'live'].index.tolist()  #  .....檢查用
# # print (cancel_test)  #檢查用
#
# #將canceled 且沒募到9成資金的直接  判定為failed 反之 successful 以募資角度確實有募到錢 使用pledged
df.loc[(df['state'] == 'canceled') & ((df['pledged'] * 1.1) < df['goal']), 'state'] = 'failed'
# # condition = (df['state'] == 'canceled') & ((df['usd_pledged'] * 1.1) < df['goal'])
# # df.loc[condition, 'state'] = 'failed'
df.loc[(df['state'] == 'canceled') & ((df['pledged'] * 1.1) >= df['goal']), 'state'] = 'successful'
#

# #做一個 state  判斷  # 定義一個函數來應用條件並返回 1 或 0
def judge(row):
    if row['pledged'] >= row['goal'] and row['state'] in ['successful', 'live']:
        return 1
    elif row['pledged'] < row['goal'] and row['state'] == 'successful':
        return 0
    elif row['pledged'] < row['goal'] and row['state'] in ['failed', 'live']:
        return 1
    elif row['pledged'] >= row['goal'] and row['state'] == 'failed':
        return 0
    elif row['state'] == 'canceled':           #因為上面已經把canceled 都變完了
        return 0
#
df['judge'] = df.apply(judge, axis=1)
filter= (df['judge'] == 0)   #此為布林值

indices_to_drop = df[filter].index
df.drop(indices_to_drop, inplace=True)
df = df.drop(df[df['state'] == 'live'].index)  #直接刪掉live

# # # # 找目標最小設美元直接算失敗
min_goal = df["goal"].min()
min_goal_indices = df.index[df['goal'] == min_goal].tolist()
print(f"min_goal: {min_goal},  min_goal_indices: {min_goal_indices}")
df.drop(min_goal_indices, inplace=True)

#新的命名
# output_file_path = '2022-06-09data/test.xlsx'
# df.to_excel(output_file_path, index=False)

#label encoding
# df['state'] = df['state'].map({"successful": 1,  "failed": 0})
# #
# # # #Onehot encoding
# # #
# onehot_encoder = OneHotEncoder()
# onehot_encoder.fit(df[["category"]])
# category_encoded = onehot_encoder.transform(df[["category"]]).toarray()
# print(category_encoded)
# #
# # # 知道讀到的順序
# categories = onehot_encoder.get_feature_names_out(["category"])
# # print(categories)
#
# df[[ "art", "comics", "crafts", "dance", "design", "fashion", "film", "food", "games",
#      "journalism",  "music", "photography","publishing", "technology", "theater" ]] = category_encoded
#
# 預試集的類別可能不滿15項
# category_data = df2[["category"]]
# # print(category_data["category"].unique())
# #
# all_categories = ["art", "comics", "crafts", "dance", "design", "fashion", "film",
#                   "food", "games", "journalism", "music", "photography", "publishing",
#                   "technology", "theater"]
# one_hot_encoded = pd.get_dummies(df2['category'])  # pd.get_dummies 好用
# print(one_hot_encoded)

# #  先用全部類別 再用現存類別 的資料去跑
# for category in all_categories:
#     if category not in one_hot_encoded.columns:
#         one_hot_encoded[category] = 0
#
# print("2:",one_hot_encoded)
# df_combined = pd.concat([df2, one_hot_encoded], axis=1)
# # print("2:",df_combined)
# df_combined = df_combined[['source_url', 'category'] + all_categories[:9] + all_categories[9:]]
# output_file_path = '2024-01-15/Newtest1.xlsx'
# df_combined.to_excel(output_file_path, index=False)


# output_file_path = '2022-06-09data/test1.xlsx'
# df.to_excel(output_file_path, index=False)


# def filter_by_zscore(df, columns, z_thresh=3):
#     """
#     计算给定列的Z分数，并过滤掉Z分数绝对值大于z_thresh的行。
#     """
#     z_scores = df[columns].apply(zscore)
#     filtered_df = df[(z_scores.abs() < z_thresh).all(axis=1)]
#     return filtered_df
#
# def print_correlation(df, col1, col2):
#     """
#     计算并打印两列之间的相关性。 .corr 已經將數據正規化了
#     """
#     correlation = df[col1].corr(df[col2])
#     print(f"Correlation between {col1} and {col2}: {correlation}")
#
# def print_spearman_correlation(df, col1, col2):
#     correlation, p_value = spearmanr(df[col1], df[col2])
#     print(f"Spearman correlation between {col1} and {col2}: {correlation}, p-value: {p_value}")
#
# # # 计算Z分数并过滤数据
# # columns_to_zscore = ['backers_count', 'usd_pledged', 'converted_pledged_amount', 'duration']
# # df_filtered = filter_by_zscore(df, columns_to_zscore, z_thresh=3)
# #
# # # 计算并打印相关性
# # print_correlation(df_filtered,'backers_count', 'usd_pledged')
# # print_correlation(df_filtered,'backers_count', 'converted_pledged_amount')
# # print_correlation(df_filtered,'duration', 'usd_pledged')
# # print_correlation(df_filtered,'duration', 'converted_pledged_amount')
# # print_correlation(df_filtered,'duration', 'backers_count')
# #
# # # 如果您还想比较直接计算相关性和使用Z分数过滤后计算相关性的差异
# print("\nDirect correlation without Z-score filtering:")
# print_correlation(df,'backers_count', 'usd_pledged')
# print_correlation(df,'backers_count', 'converted_pledged_amount')
# print_correlation(df,'duration', 'usd_pledged')
# print_correlation(df,'duration', 'converted_pledged_amount')
# print_correlation(df,'duration', 'backers_count')
#
# print("\nDirect spearman_correlation without Z-score filtering:")
# print_spearman_correlation(df,'backers_count', 'usd_pledged')
# print_spearman_correlation(df,'backers_count', 'converted_pledged_amount')
# print_spearman_correlation(df,'duration', 'usd_pledged')
# print_spearman_correlation(df,'duration', 'converted_pledged_amount')
# print_spearman_correlation(df,'duration', 'backers_count')

# plt.scatter(Xbb,Ydd)
# plt.show()