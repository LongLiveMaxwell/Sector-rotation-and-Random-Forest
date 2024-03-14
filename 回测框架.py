# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:19:08 2024

@author: LongLiveMaxwell
"""
import pandas as pd
import numpy as np
import CS2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']

import os

from sklearn.ensemble import RandomForestClassifier

def get_industries():
    '''
    

    Returns
    -------
    criteria_met_records : TYPE
        DESCRIPTION. 每个交易日行业相对集中度大于某一阈值（9），且相对估值在某一范围内的行业列表
    last_day : TYPE
        DESCRIPTION. 从2019年9月30日开始，每个月最后一个交易日

    '''
    relative_PB_ratio = pd.read_csv('./相对估值.csv')
    relative_PB_ratio.set_index('日期', inplace=True)
    relative_PB_ratio.index = pd.to_datetime(relative_PB_ratio.index)
    
    
    
    centralized_score = CS2.get_rolling_CS()
    centralized_score = centralized_score.drop('环保',axis = 1)
    centralized_score.index = pd.to_datetime(centralized_score.index)
    
    #求集中度因子相对过去2年历史数据的平均值标准差
    centralized_score_mean = centralized_score.rolling(window = 504, min_periods=1).mean()
    
    centralized_score_std = centralized_score.rolling(window = 504, min_periods=1).std()
    
    centralized_score = (centralized_score - centralized_score_mean) / centralized_score_std
    
    centralized_score = centralized_score.dropna()
    
    
    last_day = centralized_score.index.to_series().groupby([centralized_score.index.year, centralized_score.index.month]).last()
    
    #为什么从2019-09-30开始？因为2019-09-30是具体股票数据的第一个月月末交易日
    start_date = pd.to_datetime('2019-09-30')
    
    start_index = np.where(last_day == start_date)[0][0]
    
    last_day = last_day[start_index:]
    
    relative_PB_ratio = relative_PB_ratio[start_date:]
    centralized_score = centralized_score[start_date:]
    
    # 每个交易日给相对估值排序
    sorted_relative_PB_ratio = relative_PB_ratio.rank(axis=1, method='min', ascending=False)
    
    # 每个交易日给集中度因子排序（由高到低）
    sorted_centrality_score = centralized_score.rank(axis=1, method='min', ascending=False)
    
    
    
    k = 9
    
    #当集中度排名在前k+1，且相对估值在17—24，则买入
    industries_should_buy_criteria = (sorted_centrality_score <= k) & (sorted_relative_PB_ratio >= 16) & (sorted_relative_PB_ratio <= 24)
    #当集中度排名大于K+1，或者相对估值在前十，则卖出
    industries_should_sell_criteria =  (sorted_relative_PB_ratio <= 9) | (sorted_centrality_score > k)

     # 把这些行业记录下来
    criteria_should_buy = industries_should_buy_criteria.apply(lambda row: row.index[row].tolist(), axis=1)
    criteria_should_sell = industries_should_sell_criteria.apply(lambda row: row.index[row].tolist(), axis=1)

    
    return criteria_should_buy,criteria_should_sell,last_day


#predict_start_date预测期开始日，即下个月第一个交易日

def get_stock_daily_income(data,now_date,start_date,end_date,predict_start_date,predict_end_date):
    '''
    

    Parameters
    ----------
    data : Dataframe
        DESCRIPTION. 行业内所有股票在某一堆交易日内的一切数据
    now_date : TYPE
        DESCRIPTION. 当月月底的最后一天
    start_date : TYPE
        DESCRIPTION. now_date前84个交易日（相当于3个月前）
    end_date : TYPE
        DESCRIPTION. now_date前21个交易日（相当于1个月前）
    predict_start_date : TYPE
        DESCRIPTION. 预测期开始日，即下个月第一个交易日
    predict_end_date : TYPE
        DESCRIPTION. 预测期结束日，即下个月最后一个交易日

    Returns
    -------
    average_pct_change : 
        DESCRIPTION. 每个交易日选中股票的平均涨跌幅%
        
    selected_stocks : 
        DESCRIPTION. 选中股票的股票代码
    '''
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    
    
    train_data = data.loc[start_date:end_date]
    test_data = data.loc[now_date]
    
    # 去除不需要的列
    columns_to_drop = ['股票代码', '开盘价', '最高价', '最低价', '收盘价', '涨跌幅', '成交量', '未来21日后的收盘价', '未来21日后的收益率']
    X_train = train_data.drop(columns=columns_to_drop + ['label'])
    y_train = train_data['label']
    X_test = test_data.drop(columns=columns_to_drop + ['label'])

    # 训练随机森林模型
    clf.fit(X_train, y_train)
    
    
    # 对测试集进行预测，获取label==1的概率
    proba = clf.predict_proba(X_test)[:, clf.classes_ == 1].flatten()
    
    # 计算要选出的股票数量（3%）
    top_n = int(max(len(proba) * 0.03 , 2))
    
    # 获取概率最高的前5%的股票索引
    top_indices = proba.argsort()[-top_n:]
    
    # 根据索引选出股票代码
    selected_stocks = test_data.iloc[top_indices]['股票代码'].values
    

    #提取次月交易数据
    predict_data = data.loc[predict_start_date:predict_end_date]
    
    selected_stocks_df = predict_data[predict_data['股票代码'].isin(selected_stocks)]    
    
    #计算所选股票的平均涨跌幅
    average_pct_change = selected_stocks_df.groupby('日期')['涨跌幅'].mean()
    
    return average_pct_change, selected_stocks


path = "./行业内股票数据/csv文件/"

csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

all_dfs = {}

# 遍历每个CSV文件
for file_name in csv_files:
    # 从文件名中提取DataFrame的名字（去除.csv后缀）
    df_name = os.path.splitext(file_name)[0]
    
    # 读取CSV文件为DataFrame
    file_path = os.path.join(path, file_name)
    df = pd.read_csv(file_path)
    
    # 将DataFrame存储到一个字典中，以便后续访问
    all_dfs[df_name] = df
    
    

# 重新定义label
def redefine_label(label):
    if label == 5:
        return 1
    elif label == 1:
        return -1
    else:
        return 0

for file_name in csv_files:
    df_name = os.path.splitext(file_name)[0]   
    # 将日期列转换为datetime类型并设置为索引
    all_dfs[df_name]['日期'] = pd.to_datetime(all_dfs[df_name]['日期'])
    all_dfs[df_name].set_index('日期', inplace=True)
    all_dfs[df_name]['label'] = all_dfs[df_name]['label'].apply(redefine_label)



data = all_dfs['综合']

criteria_should_buy,criteria_should_sell,last_day = get_industries()

#提取所有交易日
trade_days = data.index.unique()

days = []
days.append(last_day.iloc[0])

cumulative_value = []
cumulative_value.append(1.0)

STOCKS = pd.DataFrame(columns=['日期','股票代码'])

holding_industries = set()

# 遍历每个月的最后一个交易日进行预测
for i, date in enumerate(last_day[:-1]):  
    index = np.where(trade_days == date)[0][0]
    
    start_date = trade_days[index - 84]
    end_date = trade_days[index - 21]
    
    
    #下一月的最后一日
    next_last_day = last_day.iloc[i + 1]
    
    #次月起始交易日
    predict_start_date = trade_days[index + 1]
    
    predict_end_date = next_last_day
    
    holding_industries = holding_industries - set(criteria_should_sell[date])
    
    holding_industries = holding_industries | set(criteria_should_buy[date])
    
    industries = list(holding_industries)
    
    total_stocks = []
    
    pct_change_df = pd.DataFrame()
    
    if len(industries) > 0:
        print(f"{date}选出的行业为: {industries}")
        for industry_name in industries:
            average_pct_change, selected_stocks = get_stock_daily_income(all_dfs[industry_name], date, start_date, end_date, predict_start_date, predict_end_date)
            pct_change_df = pd.concat([pct_change_df,average_pct_change],axis = 1)
            total_stocks.append(selected_stocks)
            print(f"    从行业: {industry_name} 中选取的股票代码为{selected_stocks}")
        total_average_pct_change = pct_change_df.mean(axis = 1)
        
        # 使用 append() 方法追加新行数据
        STOCKS = STOCKS.append({'日期': date, '股票代码': total_stocks}, ignore_index=True)

        prev_value = cumulative_value[-1]
        for day, value in total_average_pct_change.iteritems():
            # 计算每日的累乘结果
            cumulated = prev_value * (1 + value / 100)
            cumulative_value.append(cumulated)
            days.append(day)
            prev_value = cumulated
    else:
        for day in trade_days[(trade_days >= predict_start_date) & (trade_days <= predict_end_date)]:
            days.append(day)
            cumulative_value.append(cumulative_value[-1])
            

HS300 = pd.read_excel('./HS300.xlsx')
HS300.set_index('日期', inplace=True)
HS300 = HS300[last_day.iloc[0]:]
HS300 = HS300.div(HS300.iloc[0])

common_index = HS300.index.intersection(days)

ZZ1000 = pd.read_excel('./中证1000.xlsx')
ZZ1000.set_index('日期', inplace=True)
ZZ1000 = ZZ1000[last_day.iloc[0]:]
ZZ1000 = ZZ1000.div(ZZ1000.iloc[0])

HS300.index = pd.to_datetime(HS300.index)

ZZ1000.index = pd.to_datetime(ZZ1000.index)

HS300 = HS300.loc[common_index]
ZZ1000 = ZZ1000.loc[common_index]

graph = pd.DataFrame(columns=['累计净值','沪深300','中证1000'])
graph['沪深300'] = HS300
graph['中证1000'] = ZZ1000
graph['累计净值'] = cumulative_value

graph.plot()



