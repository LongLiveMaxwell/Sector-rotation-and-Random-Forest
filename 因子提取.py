# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 20:15:28 2024

@author: LongLiveMaxwell
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
data = pd.read_csv("./行业内股票数据/csv文件/计算机.csv")

# 定义重新定义 label 的函数
def redefine_label(label):
    #5表示收益率排前20%
    if label == 5:
        return 1
    #1表示收益率排后20%
    elif label == 1:
        return -1
    else:
        return 0

data['label'] = data['label'].apply(redefine_label)

columns_to_drop = ['日期', '股票代码', '开盘价', '最高价', '最低价', '收盘价', '涨跌幅', '成交量', '未来21日后的收盘价',
       '未来21日后的收益率']


X = data.drop(columns_to_drop + ['label'], axis=1)
y = data['label']

corr_matrix = X.corr().round(2)
plt.figure(figsize=(100,90))
sns.heatmap(corr_matrix,annot=True,linewidths=0.1,vmax=1,vmin=-1)
plt.title('Correlation Heatmaps',size=16)
plt.rcParams['axes.unicode_minus']=False    # 用来正常显示负号
plt.show()


print(corr_matrix)


sum_corr = corr_matrix.abs()
sum_corr['总相关性'] = sum_corr.sum(axis=1)
print(sum_corr)

factor_should_be_saved = ['EMA','SAR','ADX','CMO','DX','MFI','MOM','PPO','RSI','ULTOSC','ROC','WILLR','CCI','BOP','OBV','AD','ADOSC','NATR_14',
         'TRANGE','AVGPRICE','HT_DCPHASE','HT_DCPERIOD','LINEARREG_SLOPE','CORREL','LINEARREG_ANGLE','STDDEV','VAR','DIV','SIN','SUM']

X = data[factor_should_be_saved]

corr_matrix = X.corr().round(2)

import lightgbm as lgb
from sklearn.model_selection import train_test_split




# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化LightGBM模型
lgb_model = lgb.LGBMClassifier(n_estimators=200, random_state=42)

# 训练模型
lgb_model.fit(X_train, y_train)

# 获取特征重要性
feature_importances = lgb_model.feature_importances_

# 将特征重要性与特征名称组合成DataFrame
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

from lightgbm import plot_importance

Num_features = len(corr_matrix)
plot_importance(lgb_model,max_num_features = Num_features)
plt.show()


# 选择lightgbm算法的特征重要性评分前20的特征进行建模
X_train = X_train.iloc[:,features_df.index[:Num_features-2]]
X_test = X_test.iloc[:,features_df.index[:Num_features-2]]

