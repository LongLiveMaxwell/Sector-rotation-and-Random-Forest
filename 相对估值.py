# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:17:29 2024

@author: LongLiveMaxwell
"""

import pandas as pd
import matplotlib.pyplot as plt
#中文字体
import matplotlib
matplotlib.rc("font",family='DengXian')

def get_relative_PB_ratio():
    '''
    

    Returns
    -------
    relative_standardized_PB : TYPE dataframe
        DESCRIPTION.

    '''
    
    PB_ratio = pd.read_excel("D:\通往自由之路\硕士毕业论文\申万行业指数市净率.xlsx")
    
    PB_ratio.set_index('Unnamed: 0',inplace = True)
    
    #PB_ratio = PB_ratio.drop(['煤炭','美容护理','石油石化'],axis = 1)
    
    rows_with_nans = PB_ratio.isnull().any(axis = 1)
    
    
    PB_ratio = PB_ratio.interpolate(method='linear', limit_direction='both', axis=0)
    
    
    PB_ratio.plot(figsize=(30,20))
    
    rolling_average_PB_ratio = PB_ratio.rolling(window=252).mean()
    
    standardized_PB = PB_ratio / rolling_average_PB_ratio
    
    relative_standardized_PB = standardized_PB.apply(lambda x: x / (standardized_PB.drop(x.name, axis=1).mean(axis=1)))
    
    relative_standardized_PB.plot(figsize=(30,20))
    
    relative_standardized_PB.dropna().to_csv('./相对估值.csv',encoding='utf_8_sig')
    
    return relative_standardized_PB
