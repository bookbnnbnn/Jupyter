#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime, timedelta
from copy import deepcopy
from tensorflow.keras.utils import to_categorical
import talib
from talib import abstract
import random
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Nadam


# In[4]:

#　在現有的點附近生成可能的點
class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
       # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(N,i,nnarray)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1


# In[5]:


# 建立有關股票的類別
class Stock:
    stock_set = set()
    def __init__(self, stock_id, flag = 'test', date = datetime.strftime(datetime.today() - 1 * timedelta(days=365), "%Y-%m-%d"), end_date = datetime.strftime(datetime.today() , "%Y-%m-%d")):
        self.stock_id = stock_id
        self.__class__.stock_set.add(self.stock_id)
        self.data_dict = {"institutional_investors":self.institutional_investors, "price":self.price, "price_ma":self.price_ma}
        self.url = "http://api.finmindtrade.com/api/v2/data"
        self.date = date
        self.end_date = end_date
    # 抓取有關法人買賣超的資料
    def institutional_investors(self):
        parameter = {"stock_id": self.stock_id,
                     "date": self.date,
                     "end_date": self.end_date,
                     "dataset": "InstitutionalInvestorsBuySell"}
        data = requests.get(self.url, params = parameter)
        data = data.json()
        data = pd.DataFrame(data['data'])
        data.set_index('date', inplace = True)
        investment_trust = data[data['name'] == 'Investment_Trust']
        investment_trust.insert(2, 'investment_trust', investment_trust['buy'] - investment_trust['sell'])
        investment_trust = investment_trust.drop(columns = ['buy', 'name', 'sell', 'stock_id'])
        foreign_investor = data[data['name'] == 'Foreign_Investor']
        foreign_investor.insert(2, 'foreign_investor', foreign_investor['buy'] - foreign_investor['sell'])
        foreign_investor = foreign_investor.drop(columns = ['buy', 'name', 'sell', 'stock_id'])
        data = pd.merge(investment_trust, foreign_investor, on = ['date'])
        return data
    # 抓取價格
    def price(self):
        parameter = {"stock_id": self.stock_id,
             "date": self.date,
             "end_date": self.end_date,
             "dataset": "TaiwanStockPrice"}
        data = requests.get(self.url, params = parameter)
        data = data.json()
        data = pd.DataFrame(data['data'])
        data.set_index('date', inplace = True)
        data = data.drop(columns = ['Trading_money', 'spread', 'Trading_turnover', 'stock_id'])
        data.columns = ['volume', 'open', 'high', 'low', 'close']
        return data
    # 抓取價格並算出移動平均
    def price_ma(self, day):
        data = self.price()
        data[f'{day}_ma'] = data['close'].rolling(day).mean()
        return data
    # 抓取價格並找到最近9天的最大及最小值
    def max_min_price(self):
        df = self.price()
        df['close_min'] = df['close'].rolling(9).min()
        df['close_min'] = df['close_min'].shift(-4)
        df['close_max'] = df['close'].rolling(9).max()
        df['close_max'] = df['close_max'].shift(-4)
        df.loc[df.close == df.close_min, 'flag'] = 0
        df.loc[(df.close != df.close_min) & (df.close != df.close_max), 'flag'] = 1
        df.loc[df.close == df.close_max, 'flag'] = 2
        return df
    # 畫出時間與價格的關係圖並標記最近9天的最大及最小值
    def plot(self):
        df = self.max_min_price()
        buy_df = deepcopy(df)
        sell_df = deepcopy(df)
        buy_df.loc[df.flag!=0 , 'close'] = 0
        sell_df.loc[df.flag!=2 , 'close'] = 0
        plt.figure(figsize=(15, 8))
        plt.plot(df["close"])
        plt.scatter(buy_df.index, buy_df['close'], c = 'r', label = 'min in 11 days')
        plt.scatter(sell_df.index, sell_df['close'], c = 'black', label = 'max in 11 days')
        plt.ylim(df['close'].min()-10, df['close'].max()+10)
        plt.title(self.stock_id)
        plt.ylabel("price")
        plt.xlabel("date")
        date = datetime.strftime(datetime.today() - 1 * timedelta(days=365), "%Y-%m-%d")
        new_ticks = []
        for i in range(len(df.index)):
            denominator = int(len(df.index) / 10)
            new_ticks.append(df.index[i]) if i%denominator == 0 else 0
        plt.xticks(new_ticks)
        plt.legend()
    # 抓取相關指標
    def technical_index(self):
        df = self.max_min_price()
        df2 = self.institutional_investors()
        df['RSI'] = abstract.RSI(df) / 100
        df['CMO'] =(abstract.CMO(df)+100) / (2 *100)
        df['MACD'] =(abstract.MACD(df)['macd']+abstract.MACD(df)['macd'].max()) / (2 *abstract.MACD(df)['macd'].max())
        df['WILLR'] =(abstract.WILLR(df)+100) / (2 *100)
        df['WMA'] =abstract.WMA(df) / abstract.WMA(df).max()
        df['PPO'] =(abstract.PPO(df)+abstract.PPO(df).max()) / (2 *abstract.PPO(df).max())
        df['EMA'] =abstract.EMA(df) / abstract.EMA(df).max()
        df['ROC'] =(abstract.ROC(df)+abstract.ROC(df).max()) / (2 *abstract.ROC(df).max())
        df['SMA'] =abstract.SMA(df) / abstract.SMA(df).max()
        df['TEMA'] =abstract.TEMA(df) / abstract.TEMA(df).max()
        df['CCI'] =(abstract.CCI(df)+abstract.CCI(df).max()) / (2 *abstract.CCI(df).max())
        df['investment_trust'] = (df2['investment_trust'] + df2['investment_trust'].max()) / (2*df2['investment_trust'].max())
        df['foreign_investor'] = (df2['foreign_investor'] + df2['foreign_investor'].max()) / (2*df2['foreign_investor'].max())
        df = df.drop(columns=['volume', 'open', 'high', 'low', 'close', 'close_max', 'close_min'])
        df = df.dropna()
        return df


# In[6]:


# 建立交易類別且為股票的父類別
class Trade(Stock):
    def __init__(self, stock_id,flag = 'test', date = datetime.strftime(datetime.today() - 1 * timedelta(days=365), "%Y-%m-%d"), end_date = datetime.strftime(datetime.today() , "%Y-%m-%d"), balance = 1000000, fee = 0.00625, tax = 0.003, buy_flag = False, sell_flag = False):
        super().__init__(stock_id, flag, date, end_date)
        self.balance = balance
        self.fee = fee
        self.tax = tax
        self.buy_flag = buy_flag
        self.sell_flag = sell_flag
    # 進行前置作業
    def preprocessing(self):
        data1 = self.data_dict["price_ma"](10)
        data2 = self.data_dict["price_ma"](20)
        data3 = self.data_dict["institutional_investors"]()
        data = pd.merge(data1, data2, on = ['date', 'volume', 'open', 'close', 'high', 'low'], how = "inner")
        data = pd.merge(data, data3, on = ['date'], how = "inner")
        zeros = np.zeros(len(data)).reshape((len(data)), 1)
        data['buy_or_sell'] = zeros    #先將明天買賣股數設為0
        data['shares'] = zeros  #先將股數設為0
        data['balance'] = zeros  #先將餘額設為0
        data['profit'] = zeros   #先將損益設為0
        data['return'] = zeros   #先將報酬率設為0
        return data
    # 制定根據法人買賣的交易策略
    def follow_IT(self):
        df = self.preprocessing()
        df['IT_flag'] = df['investment_trust'] > 0   #投信是否買超
        df['IT_buy_days'] = df['IT_flag'].rolling(3).sum()   #投信這三天買超的天數
        #當投信連續買超3天 且 外資今天也買超 且 大於十日平均 就設為買進訊號
        df['buy_flag'] = (df['IT_buy_days'] == 3)  & (df['close'] > df['10_ma']) & (df['foreign_investor'] > 0)
        #當投信不再連續買超 或 收盤價小於十日平均 就設為賣出訊號
        df['sell_flag'] = (df['investment_trust'] < 0) & (df['close'] < df['10_ma']) & (df['foreign_investor'] < 0)
        self.buy_flag = np.array(df['buy_flag'])
        self.sell_flag = np.array(df['sell_flag'])
    # 紀錄每次的交易並計算報酬率等資訊
    def trading(self):
        df = self.preprocessing()
        df['buy_flag'] = self.buy_flag
        df['sell_flag'] = self.sell_flag
        for index, row in df.iterrows():
            #因為第一天沒有昨天的資料所以會出現Error，因此使用try
            try:
                df.loc[index, 'shares'] = yesterday.shares    #先將今天的股數設成和昨天一樣若有買賣再運算
                df.loc[index, 'balance'] = yesterday.balance  #先將今天的餘額設成和昨天一樣若有買賣再運算
                #當昨天的應買賣股數不等於0時，代表今天會執行買賣
                #(附註1：因為三大法人買賣超收盤之後才會知道因此所有動作都只能隔一天才能執行)
                if yesterday.buy_or_sell != 0 :
                    df.loc[index, 'buy_or_sell'] = 0    #將應買賣股數變回0
                    df.loc[index, 'shares'] = yesterday.shares + yesterday.buy_or_sell   #今日庫存股數為昨日股數加上應買股數
                    #今日餘額為昨日餘額加上今天交易股數乘上今天開盤價扣掉手續費和證交稅
                    df.loc[index, 'balance'] = yesterday.balance - yesterday.buy_or_sell*df.loc[index, 'open']*(1 + (self.fee + self.tax))  

                #當今天買進訊號出現的時候 且 今天的餘額夠買一張股票 明天開盤就買進
                #(附註2：應該使用明天的開盤價，不過明天還沒開盤不會知道開盤價，所以用今天收盤的漲停價來推算餘額夠不夠)
                if (df.loc[index, 'buy_flag']) and (1/2*(df.loc[index, 'balance'] - df.loc[index, 'close'] * 1.1 * 1000) > 0):  
                    shares = int(1/2*df.loc[index, 'balance'] / (df.loc[index, 'close'] * 1.1 * 1000))   #先估算餘額可以買幾股
                    df.loc[index, 'buy_or_sell'] = shares * 1000  #將預估購買的股數填入
                #當今天賣出訊號出現的時候 且 庫存股票股數大於0 明天開盤就賣出
                elif (df.loc[index, 'sell_flag']) and (df.loc[index, 'shares'] > 0):
                    df.loc[index, 'buy_or_sell'] = int(df.loc[index, 'shares'] * -1)   #將預估賣出的股數填入

                df.loc[index, 'profit'] = (df.loc[index, 'close'] * df.loc[index, 'shares'] + df.loc[index, 'balance']) - self.balance
                df.loc[index, 'return'] = round(df.loc[index, 'profit']/self.balance *100, 2)
                yesterday = df.loc[index]    #完成資料更新後將其設為下一天的昨天

            except UnboundLocalError:
                df.loc[index, 'balance'] = self.balance
                yesterday = df.loc[index]
        return df
    # 計算Sharpe Ratio、平均報酬和勝率
    def summary(self):  
        df = self.trading()
        df['last_return'] = df['return'].shift(-1)
        return_list = []
        flag = True
        for index, row in df.iterrows():
            if flag:
                if df.loc[index, 'buy_or_sell'] > 0 :
                    before_return = df.loc[index, 'return']
                    flag = False
            if flag == False:
                if df.loc[index, 'buy_or_sell'] < 0:
                    after_return = df.loc[index, 'last_return'] 
                    return_list.append(after_return - before_return)
                    flag = True
        return_arr = np.array(return_list)
        mean = return_arr.mean()
        std = np.std(return_arr, ddof = 1)
        times = len(return_arr)
        if times != 0:
            win_times = len(return_arr[return_arr > 0])
            win_rate = win_times / times
            sharpe = mean / std * (times ** 0.5)
        else:
            sharpe, win_rate = np.nan, np.nan
        return sharpe, mean, win_rate
    # 透過Line通知是否進行交易
    def line_notify(self, line_token):
        df = self.trading_folllow_IT()
        index = datetime.strftime(datetime.today(), "%Y-%m-%d")
        try:
            if df.loc[index, 'buy_or_sell'] != 0:
                flag = "買" if df.loc[date, 'buy_or_sell'] > 0 else "賣"
                msg = f"{flag} {self.stock_id} {df.loc[date, 'buy_or_sell']}股"
                stickerPackageId = 2
                stickerId = 34
                url = "https://notify-api.line.me/api/notify"
                headers = {
                    "Authorization": "Bearer " + line_token
                } 
                payload = {"message": msg, "stickerPackageId": stickerPackageId, 'stickerId': stickerId}
                r = requests.post(url, headers = headers, params = payload)
                return r.status_code
        except KeyError:
            pass


# In[7]:


# 在模型訓練前所作的前置作業
def preprocessing(stock_id, date, end_date, SOTE = False):
    df = Stock(f"{stock_id}", date = date, end_date = end_date).technical_index()
    original_x_train = [np.array(df.iloc[0+i:13+i, 1:14]).reshape(13, 13) for i in range(len(df) - 12)]
    original_x_train = np.array(original_x_train)
    df['flag'] = df['flag'].shift(-12)
    df['flag'] = df['flag'].shift(12)
    original_y_train = np.array(df['flag'].dropna())
    x_buy_train = list(original_x_train[original_y_train == 0])
    x_hold_train = list(original_x_train[original_y_train == 1])
    x_sell_train = list(original_x_train[original_y_train == 2])
    if SOTE:
        for i in sorted(np.random.choice(range(len(x_hold_train)), int(len(x_hold_train) / 5), replace=False), reverse = True):
            x_hold_train.pop(i)
        for i in np.random.choice(range(len(x_buy_train)), int(len(x_hold_train) / 2.5) - len(x_buy_train)):
            s = Smote(np.array(x_buy_train)[i],N=100)
            x_buy_train.append(s.over_sampling())
        for i in np.random.choice(range(len(x_sell_train)), int(len(x_hold_train) / 2.5) - len(x_sell_train)):
            s = Smote(np.array(x_sell_train)[i],N=100)
            x_sell_train.append(s.over_sampling())
    x_train = []
    x_train.extend(x_buy_train)
    x_train.extend(x_hold_train)
    x_train.extend(x_sell_train)
    x_train = np.array(x_train).reshape(-1, 13, 13, 1)
    y_train = [0]*len(x_buy_train) + [1]*len(x_hold_train) + [2]*len(x_sell_train)
    y_train = np.array(y_train)
    y_train = to_categorical(y_train)
    return x_train, y_train


# In[8]:


# 將不同股票的資料都放入array中
def make_data(stock_list, date, end_date, SOTE = False):
    x_train = np.array([])
    y_train = np.array([])
    for stock_id in stock_list:
        x, y = preprocessing(f"{stock_id}", date = date, end_date = end_date, SOTE = SOTE)
        x_train = np.append(x_train, x)
        y_train = np.append(y_train, y)
    x_train = x_train.reshape(-1, 13, 13, 1)
    y_train = y_train.reshape(-1, 3)
    return x_train, y_train


# In[ ]:




