# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 23:08:20 2024

@author: LongLiveMaxwell
"""

import pandas as pd
import CS
import numpy as np




rolling_CS_df = CS.get_rolling_CS()


historical_CS_mean = rolling_CS_df.rolling(window = 504, min_periods=1).mean()

historical_CS_std = rolling_CS_df.rolling(window = 504, min_periods=1).std()

rolling_CS_df = (rolling_CS_df - historical_CS_mean) / historical_CS_std

rolling_CS_df = rolling_CS_df.dropna()

rolling_CS_df.plot(figsize=(30,20))

dates_index = rolling_CS_df.index


last_day_index = []


for i in range(dates_index.size - 1):
    if(dates_index[i].month != dates_index[i + 1].month):
        last_day_index.append(dates_index[i])
        
last_day_index = last_day_index[0:95]

data = pd.read_excel(r'D:\通往自由之路\硕士毕业论文\申万行业指数.xlsx')

data.set_index('Unnamed: 0',inplace = True)

data = data.drop(['煤炭','美容护理','石油石化'],axis = 1)

month_return = data.loc[last_day_index].pct_change(1).dropna()


month_CS = rolling_CS_df.loc[last_day_index].iloc[:-1]


IC = pd.DataFrame(index=month_CS.index,columns=['IC'])
for i in range(len(month_CS.index)):
    CS = month_CS.iloc[i]
    m_return = month_return.iloc[i]
     
    ic = CS.corr(m_return)
    
    IC.iloc[i] = (ic)
    

IC.cumsum().plot()


labels=['第1分位', '第2分位', '第3分位', '第4分位', '第5分位']  

result=pd.DataFrame(index=['第1分位', '第2分位', '第3分位', '第4分位', '第5分位','多空组合']) 

result[last_day_index[0]] = 1

for i in range(1,len(last_day_index)-1):
    test_df = pd.DataFrame(index = month_CS.columns,columns=[last_day_index[i],'集中度因子','层组'])
    test_df[last_day_index[i]] = month_return.values[i]
    test_df["集中度因子"] = month_CS.values[i]
    test_df['层组']=pd.qcut(test_df['集中度因子'], 5, labels=labels) 
    a=test_df.groupby('层组').mean().iloc[:, :-1]  
    rate=pd.DataFrame(index=labels).join(a)
    rate.loc["多空组合"]=rate.loc['第5分位']-rate.loc['第1分位']
    result=result.join((1+rate).apply(lambda x: x*(result.iloc[:, -1])),how='left')
    
    
result.T.plot()


