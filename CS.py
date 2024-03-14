# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 11:57:39 2024

@author: LongLiveMaxwell
"""
import pandas as pd
import matplotlib.pyplot as plt
#中文字体
import matplotlib
matplotlib.rc("font",family='DengXian')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
    
def get_rolling_CS():
    '''
    

    Returns
    -------
    TYPE dataframe
        DESCRIPTION. 以半年(126天)为窗口期，半年为半衰期，计算资金集中度因子

    '''
    data = pd.read_excel('./申万行业指数.xlsx')
    
    values = pd.read_excel('/申万行业市值.xlsx')
    
    values = values.drop(['煤炭','美容护理','石油石化'],axis = 1)
    data = data.drop(['煤炭','美容护理','石油石化'],axis = 1)
    
    data.set_index('Unnamed: 0',inplace = True)
    
    values.set_index('Unnamed: 0',inplace = True)
    
    #收盘价变化率
    close_pctchange = data.pct_change(1).dropna()
    
    values.dropna(inplace=True)
    
    #标准化
    transfer=StandardScaler()
    close_scaled = transfer.fit_transform(close_pctchange)
    
    
    
    
    #计算各个板块的市值占比
    total_market_cap = values.sum(axis=1)
    percentage_values = values.div(total_market_cap, axis=0)
    
    #计算市值加权后的各个行业板块收益率
    weighted_df = close_pctchange.copy()
    for sector in close_pctchange.columns:
        weighted_df.loc[:,sector] = weighted_df.loc[:, sector] * percentage_values[sector]**.5
        
        
        
        
    


    def centrality_score(X, n=2):
        '''
        

        Parameters
        ----------
        X : TYPE dataframe
            DESCRIPTION. 126个交易日内的各行业收益率时间序列
        n : TYPE, optional
            DESCRIPTION. The default is 2. 主成分分析法的降维目标为2

        Returns
        -------
        C_list : TYPE
            DESCRIPTION. 当日各行业集中度因子

        '''
        #提取天数
        N = X.shape[1]
        
        
        # pca_model = PCA(n_components=n)
        # pca_model.fit(X)

        # 计算衰减因子 alpha
        halflife = 126
        alpha = 1 - np.exp(np.log(0.5) / halflife)
        
        # 生成指数衰减向量
        decay_vector = np.array([(1 - alpha)**(126-i) for i in range(126)])
        X_cov = np.cov(X.T,aweights=decay_vector)
        
        #X_cov = np.cov(X.T)
        #求特征值、特征向量
        w, v = np.linalg.eig(X_cov)
        sorted_eigenvalues = w.argsort()[::-1]
        AR = [eig/np.sum(w) for eig in w[sorted_eigenvalues]]
        EV = v.T
        
        #本来可以用pca_model直接求，但是加入了指数衰减，还是要回归PCA的本质
        # EV = pca_model.components_
        # AR = pca_model.explained_variance_ratio_
        C_list = []
        for i in range(N):
            C_num = []
            C_denom = []
            for j in range(n):
                C_num.append(AR[j] * \
                        (abs(EV[j][i])/\
                         sum([abs(EV[j][k]) for k in range(N)])))
                C_denom.append(AR[j])
                
            C_list.append(sum(C_num)/sum(C_denom))
        return C_list 
    
    rolling_Cs = [centrality_score(weighted_df.iloc[i-126:i,:]) for i in range(126, weighted_df.shape[0]+1)]
    
    # rolling_Cs = [centrality_score(close_pctchange.iloc[i-126:i,:]) for i in range(126, close_pctchange.shape[0]+1)]
    
    rolling_Cs_result = pd.DataFrame(data=rolling_Cs, columns=close_pctchange.columns, index=[date.date() for date in close_pctchange.index[125:]])

    return rolling_Cs_result