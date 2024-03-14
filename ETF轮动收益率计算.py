# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:11:43 2024

@author: LongLiveMaxwell
"""

import pandas as pd
import numpy as np
import CS2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']

shenwan_index = pd.read_excel('./申万行业指数.xlsx')
shenwan_index.set_index('Unnamed: 0', inplace=True)
shenwan_index.index = pd.to_datetime(shenwan_index.index)

shenwan_index = shenwan_index.drop(['煤炭','美容护理','石油石化','环保'],axis = 1)

shwe_index_pctchange = shenwan_index.pct_change(1)

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

centralized_score.plot(figsize=(30,20))


#提取每个月月末的最后一个交易日
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
threshold_PB_rank = len(relative_PB_ratio.columns) - k + 1

#当集中度排名在前k+1，且相对估值在17—24，则买入
industries_should_buy_criteria = (sorted_centrality_score <= k) & (sorted_relative_PB_ratio >= 16) & (sorted_relative_PB_ratio <= 24)
#当集中度排名大于K+1，或者相对估值在前十，则卖出
industries_should_sell_criteria =  (sorted_relative_PB_ratio <= 9) | (sorted_centrality_score > k)


# 把这些行业记录下来
criteria_should_buy = industries_should_buy_criteria.apply(lambda row: row.index[row].tolist(), axis=1)

criteria_should_sell = industries_should_sell_criteria.apply(lambda row: row.index[row].tolist(), axis=1)

# 累计净值，初始为1
cumulative_value = []
cumulative_value.append(1.0)

days = []
days.append(start_date)

#使用set来表示前一个月的持仓，set的运算方便进行买入卖出
holding_industries = set()


for i, date in enumerate(last_day[:-1]): 
    next_last_day = last_day.iloc[i + 1]
    
    holding_industries = holding_industries - set(criteria_should_sell[date])
    
    holding_industries = holding_industries | set(criteria_should_buy[date])
    
    industries = list(holding_industries)
    if len(industries) > 0:

        print(f"{date}选出的行业为: {industries}")
        temp = shwe_index_pctchange[date:next_last_day][industries]
        temp_mean = temp[1:].mean(axis=1)
        prev_value = cumulative_value[-1]
        for day, value in temp_mean.iteritems():
            # 计算每日的累乘结果
            cumulated = prev_value * (1 + value)
            cumulative_value.append(cumulated)
            days.append(day)
            prev_value = cumulated
    else:
        #没有行业被选中就保持前一日的净值不变
        temp = shwe_index_pctchange[date:next_last_day].index
        temp = temp[1:]
        for day in temp:
            days.append(day)
            cumulative_value.append(cumulative_value[-1])


HS300 = pd.read_excel('./HS300.xlsx')
HS300.set_index('日期', inplace=True)
HS300 = HS300[start_date:next_last_day]
HS300 = HS300.div(HS300.iloc[0])

common_index = HS300.index.intersection(days)

ZZ1000 = pd.read_excel('./中证1000.xlsx')
ZZ1000.set_index('日期', inplace=True)
ZZ1000 = ZZ1000[start_date:next_last_day]
ZZ1000 = ZZ1000.div(ZZ1000.iloc[0])

HS300.index = pd.to_datetime(HS300.index)

ZZ1000.index = pd.to_datetime(ZZ1000.index)

HS300 = HS300.loc[common_index]
ZZ1000 = ZZ1000.loc[common_index]

graph = pd.DataFrame(columns=['累计净值','沪深300','中证1000'])
graph['沪深300'] = HS300
graph['中证1000'] = ZZ1000
graph['累计净值'] = cumulative_value
graph['日期'] = days

graph.set_index('日期',inplace=True)

graph.to_csv('./行业轮动.csv',encoding='utf_8_sig')

graph.plot()
