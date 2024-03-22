import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import string
import re
from scipy.stats import zscore
from scipy.stats import spearmanr
from sklearn.preprocessing import  OneHotEncoder

directory = 'data/FinalblurbMerged_kickstarter_data.xlsx'
            #'Test_Judge.xlsx'
df = pd.read_excel(directory)

# unique_states = df['state'].unique()
# non_standard_states = [state for state in unique_states if state not in ['successful', 'failed']]
# print(f"States other than 'successful' or 'failed': {non_standard_states}")

#轉換時間 將不是數字格式的值刪除
# def process_file(file_path, start_date, encoding='ISO-8859-1'):
#     """读取并处理单个文件，返回清理后的数据。"""
#     df = pd.read_csv(file_path, encoding=encoding)
#     df.dropna(subset=['deadline', 'launched_at'], inplace=True) #確定沒有缺失值
#
#     # 转换为 datetime 并过滤日期  把不合理排除
#     df['deadline'] = pd.to_datetime(pd.to_numeric(df['deadline'], errors='coerce'), unit='s') #数值是以秒为单位的 Unix 时间戳， pd.to_datetime 轉化時間
#     df['launched_at'] = pd.to_datetime(pd.to_numeric(df['launched_at'], errors='coerce'), unit='s')
#     return df[(df['deadline'] >= start_date) & (df['launched_at'] >= start_date)]
# #
# # # #餵入兩筆不同名稱格式的值，並將其合併
# directory = '2022-06-09data/'
# start_date = pd.to_datetime('2020-01-01')
# file_paths = [os.path.join(directory, f'Kickstarter{i:03}.csv') for i in range(1, 63)] + [
#     os.path.join(directory, 'Kickstarter.csv')]
# #
# all_data = []
# for file_path in file_paths:
#     if os.path.exists(file_path):
#         df_cleaned = process_file(file_path, start_date)
#         all_data.append(df_cleaned)
#     else:
#         print(f"{file_path} not found.")
#
# merged_df = pd.concat(all_data, ignore_index=True)
# merged_df.drop_duplicates(inplace=True)    #移除重複的值
# # #
# # # # 选择指定欄位
# selected_columns = [
#     'launched_at', 'deadline', 'converted_pledged_amount',
#     'usd_pledged', 'goal', 'category','source_url',
#     'backers_count', 'state','blurb','country_displayable_name'
# ]
# final_df = merged_df[selected_columns]
# # #
# # # # 保存到新的 Excel 文件
# output_file_path = os.path.join(directory, 'FinalMerged_kickstarter_data.xlsx')
# final_df.to_excel(output_file_path, index=False)

#新增 duration 欄位
# df['launched_at'] = pd.to_datetime(df['launched_at'], format='%Y/%m/%d %I:%M:%S %p')
# df['deadline'] = pd.to_datetime(df['deadline'], format='%Y/%m/%d %I:%M:%S %p')
# df['duration'] = df['deadline'] - df['launched_at']
#
# #更新category 取 source_url內的值代替 並且將 film後面的值刪除
# df['category'] = df['source_url'].str.extract(r'/categories/([^/]+)')
# df['category'] = df['category'].str.replace(r'^film.*', 'film', regex=True)
#
# # df_filtered = df[df['state'] != 'live']                 #  .....檢查用
# # cancel_test = df[df['state'] == 'live'].index.tolist()  #  .....檢查用
# # print (cancel_test)  #檢查用
#
# #將canceled 且沒募到9成資金的直接  判定為failed 反之 successful 以募資角度確實有募到錢
# df.loc[(df['state'] == 'canceled') & ((df['usd_pledged'] * 1.1) < df['goal']), 'state'] = 'failed'
# # condition = (df['state'] == 'canceled') & ((df['usd_pledged'] * 1.1) < df['goal'])
# # df.loc[condition, 'state'] = 'failed'
# df.loc[(df['state'] == 'canceled') & ((df['usd_pledged'] * 1.1) >= df['goal']), 'state'] = 'successful'
#
# # # # 找目標最小設美元直接算失敗
# min_goal = df["goal"].min()
# min_goal_indices = df.index[df['goal'] == min_goal].tolist()
# print(f"min_goal: {min_goal},  min_goal_indices: {min_goal_indices}" )
# df.drop(min_goal_indices, inplace=True)
#
# #做一個 state  判斷  # 定義一個函數來應用條件並返回 1 或 0
# def judge(row):
#     if row['usd_pledged'] >= row['goal'] and row['state'] in ['successful', 'live']:
#         return 1
#     elif row['usd_pledged'] < row['goal'] and row['state'] == 'successful':
#         return 0
#     elif row['usd_pledged'] < row['goal'] and row['state'] in ['failed', 'live']:
#         return 1
#     elif row['usd_pledged'] >= row['goal'] and row['state'] == 'failed':
#         return 0
#     elif row['state'] == 'canceled':
#         return 1
# #
# df['judge'] = df.apply(judge, axis=1)
# filter= (df['judge'] == 0) & (df['usd_pledged']< 10)  #此為布林值
#
# indices_to_drop = df[filter].index
# df.drop(indices_to_drop, inplace=True)
# df = df.drop(df[df['state'] == 'live'].index)  #直接刪掉live
#
# output_file_path = '2022-06-09data/test.xlsx'
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
# output_file_path = '2022-06-09data/test1.xlsx'
# df.to_excel(output_file_path, index=False)

# 開始預測
# import tensorflow
# import tf
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# X = df[["backers_count","art", "comics", "crafts", "dance", "design", "fashion", "film", "food", "games",
#      "journalism",  "music", "photography","publishing", "technology", "theater"]]
# # # pandas 資料轉換 numpy
# # X_train = X_train.to_numpy()
# # X_test = X_test.to_numpy()print(df['state'].unique())
#
# Y = df["state"]
# print(Y.isna().sum())
# print(f"X: {X.shape}, Y.shape: {Y.shape}" )
# # plt.plot(X,Y,'*')
# # plt.show()
#
# model = LogisticRegression(C = 0.5, max_iter=10000 ,solver = 'newton-cg')
#
# # # random_state 必須固定 以防每次訓練數據不同
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 , random_state = 42)
# # X_train_test, X_val, Y_train_test, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2  沒必要 再做一個驗證集嗎  把test 當驗證 調參數
#
# # logregression  = LogisticRegression()
# model.fit(X_train, Y_train)
#
# y_predict = model.predict(X_test)
# print('Accuracy of logstic regression classifier on : {: .3f}'.format(model.score(X_test, Y_test)))
#
# print(confusion_matrix(Y_test,y_predict))
#
# from sklearn.metrics import classification_report
#
# print(classification_report(Y_test, y_predict))
#
# print(df['blurb'].dtypes)
# ...................................................................... 字詞分類
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import spacy
import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.gensim_models
from gensim.models import LdaModel

import warnings
warnings.filterwarnings("ignore", category= DeprecationWarning)

# nltk.download('punkt')
# df['blurb'] = df['blurb'].str.lower()
# df['blurb'] = df['blurb'].fillna('').astype(str)  #將空值轉成字串格式
#
# def remove_special(text):
#     text = text.replace('\\t', "").replace('\\n', "").replace('\\u' , " ").replace('\\', "")
#     text = text.encode('ascii', 'ignore').decode('ascii')
#     text = ' '.join(re.sub("([@#$][A-Za-z0-9]+)| (\w+:\/\/\S+)", " ", text).split())
#     return text
#
# df['blurb'] = df['blurb'].apply(remove_special)
#
# def remove_whitespace_LT(text):
#     return text.strip()
# df["blurb"] = df['blurb'].apply(remove_whitespace_LT)
#
# # 轉成數字
# # def remove_number(text):
# #     return re.sub(r"\d+", "", text)
# # df['blurb'] = df['blurb'].apply(remove_number)
# # def remove_punctuation(text):
# #     return text.translate(str.maketrans("", "", string.punctuation))  #刪除標點符號
# # # df["blurb"] = df['blurb'].apply(remove_punctuation)
# #結果會炸裂
# # def remove_whitespace_multiple(text):
# #     return re.sub('\s+', '' , text)
# # # df["blurb"] = df['blurb'].apply(remove_whitespace_multiple)
# # def remove_singlchar(text):
# #     return re.sub(r"\b[a-zA-Z]\b", "", text)
# # # df["blurb"] = df['blurb'].apply(remove_singlchar)
# #
#
# def word_tokenize_wrapper(text):
#     return word_tokenize(text)
# df["blurb"] = df['blurb'].apply(word_tokenize_wrapper)
# print("test:",df["blurb"].head())
# # # def frqDist_wrapper(text):
# # #     return FreqDist(text)
# # # df["blurb"] = df['blurb'].apply(frqDist_wrapper)
# # # print("test1:",df["blurb"].apply(lambda x : x.most_common(5)))
# #
# # # # nltk.download('stopwords')
# list_stopwords = stopwords.words('english')
# list_stopwords.extend (['a','are','as','at','be','but','by','for','if','in','into','is','it','no','not','of','on','or','such','that','the','their','then','there','these','they','this','to','was','will','and','with'])
#
# list_stopwords = set(list_stopwords)
#
# def stopwords_removal(words):
#     return [word for word in words if word not in list_stopwords]
# df["blurb"] = df["blurb"].apply(stopwords_removal)
# print("test1:",(df["blurb"].head()))
# # #
# def lemmatization(texts, allowed_postags = ["NOUN","ADJ","VERB","ADV",'PROPN'] ):
#     nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
#     #英語任務 cnn ， 在 Onenotes 上訓練
#     texts_out = []
#     for text in texts:
#         doc = nlp(text)
#         new_text = []
#         for token in doc:
#             if token.pos_ in allowed_postags:
#                 new_text.append(token.lemma_)
#         texts_out.append(" ".join(new_text))
#     return(texts_out)
# blurb_texts = df["blurb"].apply(lambda x: " ".join(x)).tolist()
# lemmatized_texts = lemmatization(blurb_texts)
#
# def gen_words(texts):
#     final = []
#     for text in texts:
#         new = gensim.utils.simple_preprocess(text, deacc= True)   #轉換成一個由小寫字母組成的單詞列表。設置 deacc=True 可以去除文本中的非字母字符
#         final.append(new)
#     return(final)
# data_words = gen_words(lemmatized_texts)
#
# id2word = corpora.Dictionary(data_words)
# corpus = []
# for text  in data_words:
#     new = id2word.doc2bow(text)
#     corpus.append(new)
# # print(corpus)
# df['blurb'] = corpus
# output_file_path = os.path.join(path, 'test2.xlsx')
# df.to_excel(output_file_path, index=False)

# lda_model = gensim.models.ldamodel.LdaModel(corpus= corpus, id2word = id2word, num_topics=30, random_state= 100, update_every= 1, chunksize=100, passes=10, alpha='auto')
#corpus : 出現次數, id2word : 單詞對應到id ,num_topics: 分幾個組合 , random_state:用於控制模型的隨機性,update_every: 指定模型在每次迭代中更新參數的文檔數,chunksize: 指定每個訓練時的文檔塊的大小

# lda_model = LdaModel(corpus, num_topics=20, id2word=id2word, passes=15)
#塞入後重跑

# use colab  run and pandas <2.0.0
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds = "mmds", R=30)
# pyLDAvis.save_html(vis, 'lda_vis.html')
# pyLDAvis.show(vis)




# X = df [["blurb",""]]

#..........................................................................開始跑分類器
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# text_data = df['blurb']
# num_topics = 20
# corpus = df['blurb']
# id2word = corpora.Dictionary(data_words)
# # lda_model = LdaModel(corpus, num_topics=num_topics, id2word=id2word, passes=15)
# print(lda_model)


# X = df[["backers_count","art", "comics", "crafts", "dance", "design", "fashion", "film", "food", "games",
#      "journalism",  "music", "photography","publishing", "technology", "theater"]]
Y = df ['state']
# print(X)
# print(Y)
# vectorizer = CountVectorizer()
# print(X)
# X = vectorizer.fit_transform(df['blurb'])

# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(df ['blurb'])
# tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
# combined_df = pd.concat([X, tfidf_df], axis=1)
# X = combined_df

# print(X)
X_train,X_test, Y_train,Y_test = train_test_split(X,Y , test_size = 0.2, random_state = 42 )
h = RandomForestRegressor(n_estimators = 100, random_state = 42)
h.fit(X_train,Y_train)
#
accuracy = h.score(X_test, Y_test)
print("模型准确率:", accuracy)
# y_pred = h.predict(X_test)
# a = mean_squared_error (Y_test,y_pred)
# print(a)

# print(df['blurb'].head())
# list_stopwords.extend([" "])
# list_stopwords = set(list_stopwords)