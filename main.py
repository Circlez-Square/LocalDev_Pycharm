import numpy as np
import pandas as pd

#直接讀取
path = '2022-06-09data/'
path1  = '2022-06-09data/'

directory = '2022-06-09data/test3_2.xlsx'
directory1 = '2022-06-09data/Newtest3_1goal.xlsx'
directory2 = '2022-06-09data/Newtest1.xlsx'
directory3 = '2022-06-09data/finaltest.xlsx'
directory4 = '2024-01-15/NewFinaltesttopic.xlsx'
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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,log_loss, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict
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

X11 = df1[["backers_count","topicc"]] # 可加goal try try
Y1 = df1["state"]

X4 = df4[["backers_count","art", "comics", "crafts", "dance", "design", "fashion", "film", "food", "games",
     "journalism",  "music", "photography","publishing", "technology", "theater",]]

# Y4 = df4["state"]

X44 = df4[["backers_count","topicc"]] # 可加goal try try

Y4 =df4["state"]
# # # print(Y.isna().sum())
# # # print(f"X: {X.shape}, Y.shape: {Y.shape}" )
# # # # plt.plot(X,Y,'*')
# # # # plt.show()
# # #
# smote = SMOTE(k_neighbors=2, random_state=42)
# X_sm, y_sm = smote.fit_resample(X11, Y1)

# ADASYN
# adasyn = ADASYN(random_state=42)
# X_ada, y_ada = adasyn.fit_resample(X2, Y2)

# BorderlineSMOTE
# bsmote = BorderlineSMOTE(random_state=42)
# X_bs, y_bs = bsmote.fit_resample(X2, Y2)
#
# # # # # # random_state 必須固定 以防每次訓練數據不同
#
# #....原始數據
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.2 , random_state = 42) #bertopic 數據
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 , random_state = 42)
# X_train, X_test, Y_train, Y_test = train_test_split(X_sm, y_sm, test_size=0.2 , random_state = 42)
#
# # print(Y_train)
#
# logregression  = LogisticRegression()
# model = LogisticRegression(C = 1, max_iter=1000, solver = 'liblinear')
# model.fit(X_train, Y_train)
#
# #隨機森林
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, Y_train)

#find xgboost 的參數
# from sklearn.model_selection import GridSearchCV
#
# param_grid = {
#     'alpha': [0.001, 0.01, 0.1, 1.0],
#     'lambda': [0.1, 0.5, 1.0, 5.0]
# }
# xgb = XGBClassifier(
#     learning_rate=0.01,
#     max_depth=6,
#     n_estimators=500,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     eval_metric='logloss'
# )
#
# grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='neg_log_loss', cv=5)
# grid_search.fit(X_train, Y_train)
#
# print(f'Best parameters found: {grid_search.best_params_}')
# print(f'Best log loss: {-grid_search.best_score_}')





#xg 參數設定
params = {
    'learning_rate': 0.01,  # 学习率
    'max_depth': 8,        # 树的最大深度
    'n_estimators': 500,   # 迭代次数
    'alpha': 0.01,        # L1 正则化
    'lambda': 0.1,         # L2 正则化
    'subsample': 0.8,      # 每次迭代的样本比例
    'colsample_bytree': 0.8,  # 每棵树使用的特征比例
    'eval_metric': 'logloss'
}

#
#
# # #xgboost 意義
xG = XGBClassifier(**params, seed = 42)
eval_set = [(X_test, Y_test)]
xG.fit(X_train, Y_train, eval_set=eval_set, verbose=True, early_stopping_rounds=20)



# # # 這是logic/XG/rf 的預測
y_predict = xG.predict(X_test)
y_pred = xG.predict(X4)
# print(y_predict)

#這是求loss值得差異  換.predict前面的模型
y_prob = xG.predict_proba(X_test)   #顯示類別的概率
loss = log_loss(Y_test, y_prob) # 訓練集loss
print(f'Log Loss: {loss:.4f}')

y_true = Y_test
#
print('Accuracy of XG regression classifier on : {: .3f}'.format(xG.score(X_test, Y_test)))
# #
print(confusion_matrix(Y4,y_pred))
cm = confusion_matrix(y_true, y_predict)
print('Confusion Matrix:',cm)

# print(classification_report(Y4, y_pred))




# rf 的loss 計算
# n_estimators_range = range(1, 101)
# log_losses = []


# for n_estimators in n_estimators_range:
#     rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
#     rf.fit(X_train, Y_train)
#     y_prob_cv = cross_val_predict(rf, X_train, Y_train, cv=3, method='predict_proba')
#     loss_cv = log_loss(Y_train, y_prob_cv)
#     log_losses.append(loss_cv)

# 绘制 log-loss 曲线
# plt.figure(figsize=(10, 6))
# plt.plot(n_estimators_range, log_losses, marker='o', color='b', label='Log Loss (Train)')
# plt.title('Random Forest Log Loss')
# plt.xlabel('Number of Trees')
# plt.ylabel('Log Loss')
# plt.legend()
# plt.grid()
# plt.show()








# # #  xgboost_loss
from sklearn.metrics import classification_report
#
print(classification_report(Y4, y_pred))
print(classification_report(y_true, y_predict))
results = xG.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(8, 6))
plt.plot(x_axis, results['validation_0']['logloss'], marker='o', label='Log Loss (Test)')
plt.title('XGBoost Log Loss')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.legend()
plt.show()


import itertools
def plot_log_loss_curve(model, X_train, Y_train, X_test, Y_test):
     losses = []
     for i in range(1, model.max_iter + 1):
          model.max_iter = i
          model.fit(X_train, Y_train)
          y_prob = model.predict_proba(X_test)
          loss = log_loss(Y_test, y_prob)
          losses.append(loss)

     plt.figure(figsize=(8, 6))
     plt.plot(range(1, model.max_iter + 1), losses, marker='o', color='b', label='Log Loss')
     plt.title('Log Loss Value')
     plt.xlabel('Iteration')
     plt.ylabel('Log Loss')
     plt.legend()
     plt.show()


# plot_log_loss_curve(model, X_train, Y_train, X_test, Y_test)
#
# print(df['blurb'].dtypes)
# ...................................................................... 字詞分類
