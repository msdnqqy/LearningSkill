import tushare as ts
import stockstats as ss
import matplotlib.pyplot as plt
data = ts.get_hist_data('600031',ktype='60')
# df = ts.get_stock_basics()
print('查询结果')
print(data)


data['date']=data.index.values
data = data.sort_index(0)
#stockStat = stockstats.StockDataFrame.retype(pd.read_csv('002032.csv'))
stockStat = ss.StockDataFrame.retype(data)
print("init finish .",stockStat.shape,stockStat)

#交易量的delta转换。交易量是正，volume_delta把跌变成负值。
stockStat[['close','close_delta']].plot(subplots=True, figsize=(20,10), grid=True)
plt.figure()
stockStat[
    ['close','cr','cr-ma1','cr-ma2','cr-ma3']
         ].plot(subplots=False, figsize=(20,10), grid=True)

stockStat[['kdjk','kdjd','kdjj'] # 分别是k d j 三个数据统计项。
         ].plot(subplots=False,figsize=(20,10), grid=True)

# MACD
stockStat[['macd','macds','macdh'] #
         ].plot(subplots=False,figsize=(20,10), grid=True)

stockStat[['close','boll','boll_ub','boll_lb'] #
         ].plot(subplots=False,figsize=(20,10), grid=True)

stockStat[['close','close_5_sma','close_10_sma'] #
         ].plot(subplots=False,figsize=(20,10), grid=True)
plt.show()
