import tensorflow as tf
import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import read_csv
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import math
def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def mape(y_true, y_pred, threshold=0.1):
    v = np.clip(np.abs(y_true), threshold, None)
    diff = np.abs((y_true - y_pred) / v)
    return 100.0 * np.mean(diff, axis=-1).mean()
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
def calc_MI(X,Y,bins):
    import numpy as np
    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI

def shan_entropy(c):
    import numpy as np
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

def generate_miss(data,perc):
    num=data.shape[0]*data.shape[1]
    list_1 = [1 for _ in range(round(num*(1-perc)))]
    list_nan = [np.nan for _ in range(round(num*perc))]
    list_1.extend(list_nan)
    random.shuffle(list_1)
    list_1=np.array(list_1).reshape(data.shape[0],data.shape[1])
    data_perc=data*list_1
    missing_mask=[]
    for i in range(data_perc.shape[0]):
        missing_mask.append(np.isnan(np.array(data_perc[i])))
    missing_mask=np.array(missing_mask).reshape(data_perc.shape[0],data_perc.shape[1])
    return data_perc
def generate_miss_3D(data,perc):#Input:samp*var*stations,perc;Output:Missing data and its mask.Eg:data,mask=generate_miss_3D(COVID_features_test,0.1)
    num=data.shape[0]*data.shape[1]*data.shape[2]
    list_1 = [1 for _ in range(round(num*(1-perc)))]
    list_nan = [np.nan for _ in range(round(num*perc))]
    list_1.extend(list_nan)
    random.shuffle(list_1)
    list_1=np.array(list_1).reshape(data.shape[0],data.shape[1],data.shape[2])
    data_perc=data*list_1
    missing_mask=[]
    for i in range(data.shape[2]):
        missing_mask.append(np.isnan(np.array(data_perc[:,:,i])))
    missing_mask=np.array(missing_mask).transpose(1,2,0)
    return data_perc,missing_mask

#根据两个点的经纬度，返回两个点的距离。
#Based on lat and long of different stations, return distance-based adjacency matrix. 
def arc2dis(LatA,LonA,LatB,LonB):
    import numpy as np
    import pandas as pd
    import math
    LatA,LatB,LonA,LonB = map(math.radians, [float(LatA), float(LatB), float(LonA), float(LonB)])
    C = math.sin((LatA-LatB)/2)* math.sin((LatA-LatB)/2)+ math.cos(LatA)*math.cos(LatB)*math.sin((LonA-LonB)/2)*math.sin((LonA-LonB)/2)
    ra=6378137
    pi=3.1415926
    dist=2*math.asin(math.sqrt(C))*6371000
    dist=round(dist/1000,3)
    return dist
def exp(a):#该函数输入列表类型的数据后，返回列表数据的指数。
    exp_a=[]
    for i in range(a.shape[0]):
        exp_a.append(math.exp(a[i]))
    return exp_a
def adj_dist(lat_arr,long_arr):#计算初始距离，用a表示
    u=[]
    for i in range(lat_arr.shape[0]):
        for j in range(long_arr.shape[0]):
            u.append(arc2dis(lat_arr.iloc[i],long_arr.iloc[i],lat_arr.iloc[j],long_arr.iloc[j]))  
    u=np.array(u)
    u=exp(-u**2/(2*(np.std(u))**2))
    u=np.array(u)
    a=u.reshape(len(lat_arr),len(long_arr))
    adj=a
    return adj
def Global_MeanMI(data1,data2,bins):
    import numpy as np
    data1=pd.DataFrame(data1)
    data2=pd.DataFrame(data2)
    Mean_MI=[]
    for i in range(data1.shape[1]):
        Mean_MI.append(calc_MI(data1.iloc[:,i],data2.iloc[:,i],bins))
    Mean_MI=np.mean(Mean_MI)
    return Mean_MI
def Global_ConcatMI(data1,data2,bins):
    data1=pd.DataFrame(data1)
    data2=pd.DataFrame(data2)
    data1=np.array(data1).reshape(-1)
    data2=np.array(data2).reshape(-1)
    Global_concatMI=calc_MI(pd.DataFrame(data1).iloc[:,0],pd.DataFrame(data2).iloc[:,0],bins)
    return Global_concatMI
def Global_EachMI(data1,data2,bins):
    import numpy as np
    data1=pd.DataFrame(data1)
    data2=pd.DataFrame(data2)
    Each_MI=[]
    for i in range(data1.shape[1]):
        Each_MI.append(calc_MI(data1.iloc[:,i],data2.iloc[:,i],bins))
    return Each_MI
def MinMax3D(data):#Input:Samp*var*station,Output:samp*var*station(after normalization)
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    data_after=[]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in range(data.shape[2]): 
        data_after.append(scaler.fit_transform(data[:,:,i]))
    data_after=np.array(data_after)
    data_after=data_after.transpose(1,2,0)#(503,3,7)
    return data_after
def Csv2Tensor(pathroot):#Input:Path root Output:samp*var*station.Eg:data=Getdire('Data/COVID/raw')(503,3,7)
    import os
    import pandas as pd
    from sklearn import linear_model
    import random
    import numpy as np
    PATH_ROOT = os.getcwd()
    ROOT = os.path.join(PATH_ROOT,pathroot)
    filenames = os.listdir(ROOT)   
    data=[]
    for i in filenames:
        PATH_CSV = os.path.join(ROOT,i)
        print(np.array(PATH_CSV).shape)
        data.append(pd.read_csv(PATH_CSV)) 
    data=np.array(data).transpose(1,2,0)#(503,3,7)
    return data

def fftTransfer1(timeseries, n=10, fmin=0.2):
    import pandas as pd
    import numpy as np
    import math
    from scipy.fftpack import fft, ifft
    import matplotlib.pyplot as plt
    import seaborn
    import scipy.signal as signal

    yf = abs(fft(timeseries))  # 取绝对值
    yfnormlize = yf / len(timeseries)  # 归一化处理
    yfhalf = yfnormlize[range(int(len(timeseries) / 2))]  # 由于对称性，只取一半区间
    yfhalf = yfhalf * 2  # y 归一化

    xf = np.arange(len(timeseries))  # 频率
    xhalf = xf[range(int(len(timeseries) / 2))]  # 取一半区间

    #     plt.subplot(212)
    #     plt.plot(xhalf, yfhalf, 'r')
    #     plt.title('FFT of Mixed wave(half side frequency range)', fontsize=10, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表

    fwbest = yfhalf[signal.argrelextrema(yfhalf, np.greater)]  # Amplitude
    xwbest = signal.argrelextrema(yfhalf, np.greater)  # Frequency
    #     plt.plot(xwbest[0][:n], fwbest[:n], 'o', c='yellow')
    #     plt.show(block=False)
    #     plt.show()

    xorder = np.argsort(-fwbest)  # 对获取到的极值进行降序排序，也就是频率越接近，越排前
    #print('xorder = ', xorder)
    xworder = list()
    xworder.append(xwbest[x] for x in xorder)  # 返回频率从大到小的极值顺序
    fworder = list()
    fworder.append(fwbest[x] for x in xorder)  # 返回幅度
    fwbest = fwbest[fwbest >= fmin].copy()
    x=len(timeseries) / xwbest[0][:len(fwbest)]
    y=fwbest
    a=np.zeros((len(y),2), dtype='float32')
    a[:,0]=y#Get amplitude
    a[:,1]=x#Get Periodic terms
    df=pd.DataFrame(a)
    df.set_axis(["amp","period"],axis=1,inplace=True)
    # sorting data frame by name
    df.sort_values("amp", axis = 0, ascending = False,
                 inplace = True, na_position ='last')
    df=df.iloc[:n,:]
    return df

