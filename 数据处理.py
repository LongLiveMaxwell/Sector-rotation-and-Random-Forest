# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:25:07 2024

@author: LongLiveMaxwell
"""

import pandas as pd

import talib as tl

 

filelist = ['有色金属', '建筑装饰','农林牧渔', '家用电器', '社会服务', '国防军工', '交通运输', '建筑材料', '电力设备', 
       '非银金融', '基础化工', '医药生物', '商贸零售', '公用事业', '轻工制造', '机械设备', '纺织服饰', '食品饮料',
       '计算机', '房地产', '通信', '传媒', '钢铁', '综合', '银行', '汽车', '环保', '电子']

for file_name in filelist:
    path = "./行业内股票数据/" + file_name + ".xlsx"
    data = pd.read_excel(path)
    
    
    transfer = pd.DataFrame(columns=['日期','股票代码','开盘价','最高价','最低价','收盘价','涨跌幅','成交量'])
    
    dates = data.iloc[1:,0]
    
    for i in range(1, data.shape[1] ,6):
        
        #要是第一天开盘价为0，证明最早那一天股票还没上市，直接跳过
        if data.iloc[1,i] == 0:
            continue
        temp_df = pd.DataFrame(columns=['日期','股票代码','开盘价','最高价','最低价','收盘价','成交量'])
        temp_df['日期'] = dates
        temp_df['股票代码'] = data.columns[i]
        temp_df['开盘价'] = data.iloc[1:,i].replace(0,method='ffill')
        temp_df['最高价'] = data.iloc[1:,i+1].replace(0,method='ffill')
        temp_df['最低价'] = data.iloc[1:,i+2].replace(0,method='ffill')
        temp_df['收盘价'] = data.iloc[1:,i+3].replace(0,method='ffill')
        temp_df['涨跌幅'] = data.iloc[1:,i+4]
        temp_df['成交量'] = data.iloc[1:,i+5]
        temp_df['未来21日后的收盘价'] = temp_df['收盘价'].shift(-21)
        temp_df['未来21日后的收益率'] = (temp_df['未来21日后的收盘价'] / temp_df['收盘价'] - 1).astype(float)
        # 计算Overlap Studies Functions重叠指标
        # 1.EMA指数平均数
        temp_df['EMA'] = tl.EMA(temp_df['收盘价'], timeperiod=30)
        # 2.DEMA双移动平均线
        temp_df['DEMA'] = tl.DEMA(temp_df['收盘价'], timeperiod=30)
        # 3.WMA移动加权平均法
        temp_df['WMA'] = tl.WMA(temp_df['收盘价'], timeperiod=30)
        # 4.KAMA考夫曼的自适应移动平均线
        temp_df['KAMA'] = tl.KAMA(temp_df['收盘价'], timeperiod=30)
        # 5.MA移动平均线
        temp_df['MA'] = tl.MA(temp_df['收盘价'], timeperiod=30, matype=0)
        # 6.SAR抛物线指标
        temp_df['SAR'] = tl.SAR(temp_df['最高价'], temp_df['最低价'], acceleration=0, maximum=0)
        # 7.SMA简单移动平均线
        temp_df['SMA'] = tl.SMA(temp_df['收盘价'], timeperiod=30)
        # 8.T3三重指数移动平均线
        temp_df['T3'] = tl.T3(temp_df['收盘价'], timeperiod=5, vfactor=0)
    
    
        # Momentum Indicator Functions动量指标
        # 1.ADX平均趋向指数
        temp_df['ADX'] = tl.ADX(temp_df['最高价'], temp_df['最低价'], temp_df['收盘价'], timeperiod=14)
        # 2.CMO钱德动量摆动指标
        temp_df['CMO'] = tl.CMO(temp_df['收盘价'], timeperiod=14)
        # 3.DX动向指标或趋向指标
        temp_df['DX'] = tl.DX(temp_df['最高价'], temp_df['最低价'], temp_df['收盘价'], timeperiod=14)
        # 4.MFI资金流量指标
        temp_df['MFI'] = tl.MFI(temp_df['最高价'], temp_df['最低价'], temp_df['收盘价'],temp_df['成交量'], timeperiod=14)
        # 5.MOM动量
        temp_df['MOM'] = tl.MOM(temp_df['收盘价'], timeperiod=10)
        # 6.PPO价格震荡百分比指数
        temp_df['PPO'] = tl.PPO(temp_df['收盘价'], fastperiod=12, slowperiod=26, matype=0)
        # 7.RSI相对强弱指数
        temp_df['RSI'] = tl.RSI(temp_df['收盘价'], timeperiod=14)
        # 8.ULTOSC终极波动指标
        temp_df['ULTOSC'] = tl.ULTOSC(temp_df['最高价'], temp_df['最低价'], temp_df['收盘价'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
        # 9.WILLR威廉指标
        temp_df['WILLR'] = tl.WILLR(temp_df['最高价'], temp_df['最低价'], temp_df['收盘价'], timeperiod=14)
        # 10.CCI顺势指标
        temp_df['CCI'] = tl.CCI(temp_df['最高价'], temp_df['最低价'], temp_df['收盘价'], timeperiod=14)
        # 11.ROC变更率指标
        temp_df['ROC'] = tl.ROC(temp_df['收盘价'], timeperiod=10)
        # 12.BOP均势指标
        temp_df['BOP'] = tl.BOP(temp_df['开盘价'],temp_df['最高价'], temp_df['最低价'], temp_df['收盘价'])
    
    
    
        # Volume Indicators成交量指标
        # 1.AD累积/派发线
        temp_df['AD'] = tl.AD(temp_df['最高价'], temp_df['最低价'], temp_df['收盘价'],temp_df['成交量'])
        # 2.OBV能量潮
        temp_df['OBV'] = tl.OBV(temp_df['收盘价'],temp_df['成交量'])
        # 3.ADOSC震荡指标
        temp_df['ADOSC'] = tl.ADOSC(temp_df['最高价'], temp_df['最低价'],temp_df['收盘价'],temp_df['成交量'],fastperiod=3, slowperiod=10)
    
    
    
        # Volatility Indicator Functions波动率指标函数
        # 1.ATR真实波动幅度均值
        temp_df['ATR'] = tl.ATR(temp_df['最高价'], temp_df['最低价'], temp_df['收盘价'], timeperiod=14)
        # 2.NATR归一化波动幅度均值
        temp_df['NATR_14'] = tl.NATR(temp_df['最高价'], temp_df['最低价'], temp_df['收盘价'], timeperiod=14)
        # 3.TRANGE真正的范围
        temp_df['TRANGE'] = tl.TRANGE(temp_df['最高价'], temp_df['最低价'], temp_df['收盘价'])
    
    
    
        # Price Transform Functions价格指标
        # 1.AVGPRICE平均价格函数
        temp_df['AVGPRICE'] = tl.AVGPRICE(temp_df['开盘价'], temp_df['最高价'], temp_df['最低价'], temp_df['收盘价'])
        # 2.MEDPRICE中位数价格
        temp_df['MEDPRICE'] = tl.MEDPRICE(temp_df['最高价'], temp_df['最低价'])
        # 3.TYPPRICE表明性价格
        temp_df['TYPPRICE'] = tl.TYPPRICE(temp_df['最高价'], temp_df['最低价'], temp_df['收盘价'])
        # 4.WCLPRICE加权收盘价
        temp_df['WCLPRICE'] = tl.WCLPRICE(temp_df['最高价'], temp_df['最低价'], temp_df['收盘价'])
    
    
    
        # Cycle Indicator Functions周期指标
        # 1.HT_DCPHASE希尔伯特变换-主导循环阶段
        temp_df['HT_DCPHASE'] = tl.HT_DCPHASE(temp_df['收盘价'])
        # 2.HT_DCPERIOD希尔伯特变换-主导周期
        temp_df['HT_DCPERIOD'] = tl.HT_DCPERIOD(temp_df['收盘价'])
        # 3.HT_TRENDMODE希尔伯特变换-趋势与周期模式
        temp_df['HT_TRENDMODE'] = tl.HT_TRENDMODE(temp_df['收盘价'])
    
    
    
        # Statistic Functions 统计类指标
        # 1.LINEARREG_SLOPE
        temp_df['LINEARREG_SLOPE'] = tl.LINEARREG_SLOPE(temp_df['收盘价'], timeperiod=14)
        # 2.BATE贝塔系数
        temp_df['BETA'] = tl.BETA(temp_df['最高价'], temp_df['最低价'], timeperiod=5)
        # 3.LINEARREG线性回归
        temp_df['LINEARREG'] = tl.LINEARREG(temp_df['收盘价'], timeperiod=14)
        # 4.CORREL皮尔逊相关系数
        temp_df['CORREL'] = tl.CORREL(temp_df['最高价'], temp_df['最低价'], timeperiod=30)
        # 5.LINEARREG_ANGLE线性回归的角度
        temp_df['LINEARREG_ANGLE'] = tl.LINEARREG_ANGLE(temp_df['收盘价'], timeperiod=14)
        # 6.LINEARREG_INTERCEPT线性回归截距
        temp_df['LINEARREG_INTERCEPT'] = tl.LINEARREG_INTERCEPT(temp_df['收盘价'], timeperiod=14)
        # 7.STDDEV标准偏差
        temp_df['STDDEV'] = tl.STDDEV(temp_df['最低价'], timeperiod=5, nbdev=1)
        # 8.TSF时间序列预测
        temp_df['TSF'] = tl.TSF(temp_df['最低价'], timeperiod=14)
        # 9.VAR方差
        temp_df['VAR'] = tl.VAR(temp_df['最低价'], timeperiod=5, nbdev=1)
    
    
    
    
        # Math Operator Functions 数学方法
        # 1.ADD向量加法运算
        temp_df['ADD'] = tl.ADD(temp_df['最高价'], temp_df['最低价'])
        # 2.DIV向量除法运算
        temp_df['DIV'] = tl.DIV(temp_df['最高价'], temp_df['最低价'])
        # 3.SUM周期内求和
        temp_df['SUM'] = tl.SUM(temp_df['收盘价'], timeperiod=30)
        
        
        # Math Transform 数学变换
        # 1.反余弦函数，三角函数
        temp_df['COS'] = tl.COS(temp_df['收盘价'])
        # 2.反正弦函数，三角函数
        temp_df['SIN'] = tl.SIN(temp_df['收盘价'])
        # 3.数字的反正切值，三角函数
        temp_df['TAN'] = tl.TAN(temp_df['收盘价'])
        
        transfer = pd.concat([transfer, temp_df], axis=0)
        
    
    transfer = transfer.sort_values(by=['日期','股票代码'])
    
    transfer.set_index('日期',inplace = True)
    
    transfer = transfer.dropna()
    
    def label_sort(group):
        #把未来21日收益率进行排序，由小到大分为5组
        #其实本来想用随机森林模型进行多分类，用1—5评判未来21日收益率在哪个区间，但是后来发现准确率不高，
        #所以改成了后面前20%定义为1，后20%定义为-1
        group['label'] = pd.qcut(group['未来21日后的收益率'], q=5, labels=False) + 1
        return group
    
    transfer = transfer.groupby('日期').apply(label_sort)

    transfer.to_csv("./行业内股票数据/csv文件/" + file_name + ".csv",encoding='utf_8_sig')

