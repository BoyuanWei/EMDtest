import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import readcsv
from pyramid.arima import auto_arima
import matplotlib.pyplot as plt
from plotcheck import pl
plt.style.use('fivethirtyeight')
import scipy.io as scio
path = "/home/bwei/PycharmProjects/data lib/pvtotal.csv"
def hour(data):
    days=int(np.floor(len(data)/12))
    data_new = []
    for n in np.arange(days):
        data_new.append(sum(data[n*12:(n+1)*12]))
    return np.array(data_new)
#windset = rd(path)
#name = raw_input('the name of data set?')
#realwindset = windset[name]
pointsperday = 288 # CHANGE HERE FOR DIFFERENT RESOLUTION
realwindset = readcsv.rd(path)
windsetoriginal= realwindset
realwindset.shape = (len(realwindset),)
cut = input('from where the rest will be testing set?<'+str(len(realwindset)))
madataset = realwindset[:cut]
data_raw = hour(madataset)
data = data_raw[:-24]
test = data_raw[-24:]
#scio.savemat('/home/bwei/PycharmProjects/EMDtest/total.mat', {'madataset':madataset})
# Generate all different combinations of p, q and q triplets
stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=6, max_q=6, m=24,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
future_forecast = stepwise_model.predict(n_periods=24)
x = np.arange(24)
fig = plt.figure(figsize=(20, 10))
plt.plot(x,future_forecast)
plt.plot(x,test)
plt.legend
plt.show()




