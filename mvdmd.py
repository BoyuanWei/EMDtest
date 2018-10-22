# moving average on historical data and then do the dmd, hope it works
import matplotlib.pyplot as plt
from readdata import read as rd
from PyEMD import EMD
from termcolor import *
from pydmd import HODMD
import numpy as np
import readcsv
from math import sqrt
from evaluation import ev as ev
from evaluation import randomextract as re
import termcolor
from plotcheck import pl

path = "/home/bwei/PycharmProjects/data lib/pvtotal.csv"
#windset = rd(path)
#name = raw_input('the name of data set?')
#realwindset = windset[name]
pointsperday = 288 # CHANGE HERE FOR DIFFERENT RESOLUTION
realwindset = readcsv.rd(path)
windsetoriginal= realwindset
realwindset.shape = (len(realwindset),)
#data reading is done
mvamode = raw_input('which kind of MA? ema or custom?')
if mvamode == 'ema':
    def ma(data, days_to_keep, alpha=0.2, points_per_day=288):
        days_covered = int(np.floor(len(data)/points_per_day))
        points_covered = days_covered*points_per_day
        daysdata = []
        onedaydata = data[len(data)-points_covered:len(data)-points_covered+points_per_day]
        for loop in np.arange(days_covered-1):
            onedaydata = onedaydata*(1-alpha)+alpha*data[len(data)-points_covered+points_per_day*(loop+1):
                                                         len(data)-points_covered+points_per_day*(loop+1)+points_per_day]
            daysdata.append(onedaydata)
        return daysdata[-days_to_keep:], daysdata
else:
    pass#to be done

#moving average definition is done
days = input('how many days will be generated for forecasting?')
cut = input('from where the rest will be testing set?<'+str(len(realwindset)))
madataset = realwindset[:cut]
maresult = ma(madataset, days)
dmddataset_org = maresult[0]
dmddataset = []
for n in np.arange(len(dmddataset_org)):
    dmddataset.extend(dmddataset_org[n])
dmddataset = np.array(dmddataset)
hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=288).fit(dmddataset)
hodmd.reconstructed_data.shape
hodmd.plot_eigs()
hodmd.dmd_time['tend'] = (days+1)*pointsperday

#----one pic about the with DMD one
fig = plt.figure(figsize=(15, 5))
plt.plot(hodmd.original_timesteps+cut-days*pointsperday, dmddataset, '.', label='snapshots')
data_practical = windsetoriginal[cut+hodmd.dmd_timesteps-days*pointsperday]
data_prediction = hodmd.reconstructed_data[0].real
#----some special correction on the decomposition-----
zeroindex = np.where(dmddataset_org[-1] <= 0.01)[0]
data_prediction[days*pointsperday+zeroindex] = dmddataset_org[-1][zeroindex]
#-----special correction is done--------------
plt.plot(cut+hodmd.dmd_timesteps-days*pointsperday, data_practical, '-', label='the practical signal', color='g')
plt.plot(cut+hodmd.dmd_timesteps-days*pointsperday, data_prediction, '--', label='DMD output', color='r')
plt.vlines(cut, 0, 20, colors="black", linestyles="--")
plt.legend()
plt.show()

#-----another pic on without DMD-----

plt.figure(figsize=(15, 5))
#plt.plot(hodmd.original_timesteps+cut-days*pointsperday, dmddataset, '.', label='snapshots')
plt.plot(cut+hodmd.dmd_timesteps-days*pointsperday, data_practical, '-', label='the practical signal', color='g')
plt.vlines(cut, 0, 20, colors="black", linestyles="--")
plt.plot(cut+np.linspace(1, pointsperday, pointsperday, dtype='int'), dmddataset_org[-1], '--', label='ema simple forecast', color='r')
plt.legend()
plt.show()

#---just some testing pic-----
plt.figure(figsize=(15, 5))
plt.plot(cut+hodmd.dmd_timesteps-days*pointsperday, data_practical-data_prediction, '--', label='DMD output', color='r')
plt.vlines(cut, 0, 20, colors="black", linestyles="--")
plt.legend()
plt.show()




#Start the error evaluation section from here(:
print(colored('------Error evaluation for with DMD:------', 'green'))
data_prediction_eva = data_prediction[-pointsperday:]
data_practical_eva = data_practical[-pointsperday:]
ev_result = ev(data_prediction_eva, data_practical_eva)
for key in ev_result:
    print '%s: %s' % (key, ev_result[key])

#Start the error evaluation section from here(:
print(colored('------Error evaluation for without DMD:------', 'green'))
data_prediction_eva = dmddataset_org[-1]
ev_result = ev(data_prediction_eva, data_practical_eva)
for key in ev_result:
    print '%s: %s' % (key, ev_result[key])

differset = re(madataset, maresult[1])
