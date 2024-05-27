import numpy as np
import pandas as pd

#直接讀取
path = '2022-06-09data/'
path1  = '2022-06-09data/'

directory = '2022-06-09data/test3_2.xlsx'
directory1 = '2022-06-09data/finaltest1.xlsx'
directory2 = '2022-06-09data/Newtest1.xlsx'
directory3 = '2022-06-09data/finaltest.xlsx'
directory4 = '2024-01-15/Newtest3goal.xlsx'
# 'FinalcountryMerged_kickstarter_data.xlsx' FinalMerged_kickstarter_data.xlsx
df = pd.read_excel(directory)
df1 = pd.read_excel(directory1)   #預測資料
df2 = pd.read_excel(directory2)  #6萬多筆資料
df3 = pd.read_excel(directory3)
df4 = pd.read_excel(directory4)
print("okokokokokokokokokokok")

# 開始預測
import tensorflow
import tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
# pandas 資料轉換 numpy
# # X_train = X_train.to_numpy()
# # X_test = X_test.to_numpy()print(df['state'].unique()) 檢查

X2 = df2[["backers_count","art", "comics", "crafts", "dance", "design", "fashion", "film", "food", "games",
     "journalism",  "music", "photography","publishing", "technology", "theater"]]

Y2 = df2["state"]

X1 = df1[["backers_count","art", "comics", "crafts", "dance", "design", "fashion", "film", "food", "games",
     "journalism",  "music", "photography","publishing", "technology", "theater"]]

X11 = df1[["backers_count","topicc"]]
Y1 = df1["state"]

X4 = df4[["backers_count","art", "comics", "crafts", "dance", "design", "fashion", "film", "food", "games",
     "journalism",  "music", "photography","publishing", "technology", "theater"]]

Y4 = df4["state"]

# X2 = df2[["backers_count","topicc"]]
#
# Y2 =df2["state"]
# # # print(Y.isna().sum())
# # # print(f"X: {X.shape}, Y.shape: {Y.shape}" )
# # # # plt.plot(X,Y,'*')
# # # # plt.show()
# # #
# smote = SMOTE(random_state=42)
# X_sm, y_sm = smote.fit_resample(X2, Y2)

# ADASYN
# adasyn = ADASYN(random_state=42)
# X_ada, y_ada = adasyn.fit_resample(X2, Y2)

# BorderlineSMOTE
bsmote = BorderlineSMOTE(random_state=42)
X_bs, y_bs = bsmote.fit_resample(X2, Y2)

# # # # # random_state 必須固定 以防每次訓練數據不同

#....原始數據
# X_train, X_test, Y_train, Y_test = train_test_split(X2, Y2, test_size=0.2 , random_state = 42) #bertopic 數據
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 , random_state = 42)
X_train, X_test, Y_train, Y_test = train_test_split(X_bs, y_bs, test_size=0.2 , random_state = 42)

# print(Y_train)

# logregression  = LogisticRegression()
# model = LogisticRegression(C = 0.01, max_iter=10000, solver = 'liblinear')
# model.fit(X_train, Y_train)

#隨機森林
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train, Y_train)

#xgboost 意義
xG = XGBClassifier(seed = 42)
xG.fit(X_train, Y_train)
# # 這是logic 的預測
# y_predict = model.predict(X_test)
# y_pred = model.predict(X11)
# print(y_predict)


y_predict = xG.predict(X_test)
y_pred= xG.predict(X4)  #餵入的數值記得改
y_true = Y_test

print('Accuracy of logstic regression classifier on : {: .3f}'.format(xG.score(X_test, Y_test)))
# #
# print(confusion_matrix(Y1,y_pred))
cm = confusion_matrix(y_true, y_predict)
print('Confusion Matrix:',cm)


# #
from sklearn.metrics import classification_report
#
print(classification_report(Y4, y_pred))
print(classification_report(y_true, y_predict))


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
# warnings.filterwarnings("ignore", category= DeprecationWarning)
#
# # nltk.download('punkt')
# df['blurb'] = df['blurb'].str.lower()
# df['blurb'] = df['blurb'].fillna('').astype(str)  #將空值轉成字串格式
#
# df3['blurb'] = df3['blurb'].str.lower()
# df3['blurb'] = df3['blurb'].fillna('').astype(str)
# #
# def remove_special(text):
#     text = text.replace('\\t', "").replace('\\n', "").replace('\\u' , " ").replace('\\', "")
#     text = text.encode('ascii', 'ignore').decode('ascii')
#     text = ' '.join(re.sub("([@#$][A-Za-z0-9]+)| (\w+:\/\/\S+)", " ", text).split())
#     return text
#
# df['blurb'] = df['blurb'].apply(remove_special)
# df3['blurb'] = df3['blurb'].apply(remove_special)
#
# #
# def remove_whitespace_LT(text):
#     return text.strip()
# df["blurb"] = df['blurb'].apply(remove_whitespace_LT)
# df3["blurb"] = df3['blurb'].apply(remove_whitespace_LT)
# with open('blurb.pickle', 'wb') as f:
#     pickle.dump(df["blurb"], f)

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
# X = df[["backers_count","topic"]]
# Y = df ['state']
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
# X_train,X_test, Y_train,Y_test = train_test_split(X,Y , test_size = 0.2, random_state = 42 )
# h = RandomForestRegressor(n_estimators = 100, random_state = 42)
# h.fit(X_train,Y_train)
# # #
# accuracy = h.score(X_test, Y_test)
# print("模型准确率:", accuracy)
# y_pred = h.predict(X_test)
# a = mean_squared_error (Y_test,y_pred)
# print(a)

# print(df['blurb'].head())
# list_stopwords.extend([" "])
# list_stopwords = set


# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import os
import torch
import keras
import joblib
#
from bertopic import BERTopic
from openai import OpenAI
from hdbscan import HDBSCAN

np.random.seed(42)
topic_model = BERTopic(embedding_model = "all-MiniLM-L6-v2",nr_topics= 900)
topics , probs = topic_model.fit_transform(df2["blurb"])  #需搭配文字的前處理

joblib.dump(topic_model, 'topic_model.pkl') #將topic_model 保留進去名為 topic_model.pkl 的文件內
topic_model = joblib.load('topic_model.pkl')

new_topics, new_probs = topic_model.transform(df2["blurb"])
new_topicss, new_probss = topic_model.transform(df4["blurb"])
print("yes")

# 測試模型是否順利保存
# for i in range(5):
#     print(f"Iteration {i+1} for df:")
#     print(topic_model.get_topic_info())
#     print("-"*50)
#
# # 對於 df3 數據集
# for i in range(5):
#     print(f"Iteration {i+1} for df3:")
#     print(topic_model.get_topic_info())
#     print("-"*50)


pd.set_option('display.max_rows', None)
# new_topics_list = [t + 1 for t in new_topics]
new_topics = [max(0, t) for t in new_topics]
df4_with_topics = pd.DataFrame({'blurb': df3["blurb"], 'topic': new_topicss, 'topic_probss': list(new_probss)})



# 將新的topic寫入
df['topicc'] = new_topics
df['topic_probss'] = list(new_probs)

# 將結果寫入df3表格
df4['topicc'] = new_topicss
df4['topic_probss'] = list(new_probss)

output_file_path = os.path.join(path, 'Finaltesttopic.xlsx')
output_file_path1 = os.path.join(path, 'test3_1.xlsx')
df3.to_excel(output_file_path, index=False)
df.to_excel(output_file_path1, index=False)
print(topic_model.get_topic_info())