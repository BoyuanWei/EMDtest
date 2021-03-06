# moving average on historical data and then do the dmd, hope it works
import matplotlib.pyplot as plt
from readdata import read as rd
from termcolor import *
from pydmd import HODMD
import numpy as np
import readcsv
from math import sqrt
from evaluation import ev as ev
from evaluation import randomextract as re
from evaluation import pointprediction as pp
from evaluation import dmddiffer as dd
from evaluation import egp
from evaluation import gp_prediction as gpp
from evaluation import gaojier
from evaluation import datawash
from evaluation import directions as di
from evaluation import percents
from evaluation import hour
from evaluation import gp
import termcolor
from plotcheck import pl
from evaluation import baynetwork
from evaluation import bino
from evaluation import bayesianstatistic as bs
from evaluation import bsbino as bsb
from evaluation import emddmd
from evaluation import emddmdp
from evaluation import firstone
from evaluation import getfirstone
from evaluation import baydmd

path = "/home/bwei/PycharmProjects/data lib/PVhourly6months.csv"
#windset = rd(path)
#name = raw_input('the name of data set?')
#realwindset = windset[name]
pointsperday = 24 # CHANGE HERE FOR DIFFERENT RESOLUTION
realwindset = readcsv.rd(path)
#realwindset=hour(realwindset)#############
windsetoriginal= realwindset
realwindset.shape = (len(realwindset),)
#data reading is done
#mvamode = raw_input('which kind of MA? ema or custom?')
#if mvamode == 'ema':
def ma(data, days_to_keep, points_per_day, alpha=0.2 ):
    days_covered = int(np.floor(len(data)/points_per_day))
    points_covered = days_covered*points_per_day
    daysdata = []
    onedaydata = data[len(data)-points_covered:len(data)-points_covered+points_per_day]
    for loop in np.arange(days_covered-1):
        onedaydata = onedaydata*(1-alpha)+alpha*data[len(data)-points_covered+points_per_day*(loop+1):
                                                         len(data)-points_covered+points_per_day*(loop+1)+points_per_day]
        daysdata.append(onedaydata)
    return daysdata[-days_to_keep:], daysdata
#else:
    #pass#to be done

#moving average definition is done
days = input('how many days will be generated for forecasting?')
cut = input('from where the rest will be testing set?<'+str(len(realwindset)))
madataset = realwindset[:cut]
maresult = ma(madataset, days, pointsperday)
dmddataset_org = maresult[0]
dmddataset = []
for n in np.arange(len(dmddataset_org)):
    dmddataset.extend(dmddataset_org[n])
dmddataset = np.array(dmddataset)
hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=pointsperday).fit(dmddataset)
hodmd.reconstructed_data.shape
hodmd.plot_eigs()
hodmd.dmd_time['tend'] = (days+1)*pointsperday-1# since it starts from zero

#----one pic about the with DMD one
fig = plt.figure(figsize=(20, 10))
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

#plt.figure(figsize=(20, 10))
#plt.plot(hodmd.original_timesteps+cut-days*pointsperday, dmddataset, '.', label='snapshots')
#plt.plot(cut+hodmd.dmd_timesteps-days*pointsperday, data_practical, '-', label='the practical signal', color='g')
#plt.vlines(cut, 0, 20, colors="black", linestyles="--")
#plt.plot(cut+np.linspace(1, pointsperday, pointsperday, dtype='int'), dmddataset_org[-1], '--', label='ema simple forecast', color='r')
#plt.legend()
#plt.show()

#---just some testing pic-----
#plt.figure(figsize=(15, 5))
#testdata = data_practical-data_prediction
#plt.plot(cut+hodmd.dmd_timesteps-days*pointsperday, data_practical-data_prediction, '--', label='DMD output', color='r')
#plt.vlines(cut, 0, 20, colors="black", linestyles="--")
#plt.legend()
#plt.show()




#Start the error evaluation section from here(:
print(colored('------Error evaluation for with DMD:------', 'green'))
data_prediction_dmd = data_prediction[-pointsperday:]
data_practical_eva = data_practical[-pointsperday:]
ev_result = ev(data_prediction_dmd, data_practical_eva)
for key in ev_result:
    print '%s: %s' % (key, ev_result[key])

#Start the error evaluation section from here(:
print(colored('------Error evaluation for without DMD:------', 'green'))
data_prediction_ema = dmddataset_org[-1]
ev_result = ev(data_prediction_ema, data_practical_eva)
for key in ev_result:
    print '%s: %s' % (key, ev_result[key])

dmd_difference_set = dd(maresult[1], days)
differset = re(madataset, dmd_difference_set, points_per_day=pointsperday) # calculate the differences between ema results and practical data
differset = datawash(differset)
debug_flag = 555 # set this to another number if want to fix the error forecast algorithm
if debug_flag == 2: #rate mode
    bb = baynetwork(differset, dmddataset_org[-1])
    error_forecast = np.array(bino(bb, differset, dmddataset_org[-1]))
    realerror = data_practical[-pointsperday:]-data_prediction[-pointsperday:]
    succ_rate = percents(error_forecast, realerror, dmddataset_org[-1])
    print 'the success rate is: %s' % succ_rate[0]
    print 'the action rate is: %s' % succ_rate[1]

if debug_flag == 0:
    #error_forecast = np.array(pp(differset)[0]) # get the error forecast
    #error_forecast = di(error_forecast)
    #bb = bs(differset, dmddataset_org[-1])
    #error_forecast = np.array(bsb(bb, differset, dmddataset_org[-1]))
    error_forecast = emddmdp(differset, dmddataset_org[-1])
    #error_forecast = baydmd(differset, dmddataset_org[-1])
    fig = plt.figure(figsize=(20, 10))
    plt.plot(hodmd.original_timesteps + cut - days * pointsperday, dmddataset, '.', label='snapshots')
    data_practical = windsetoriginal[cut + hodmd.dmd_timesteps - days * pointsperday]
    data_prediction = hodmd.reconstructed_data[0].real
    data_predicion_org = hodmd.reconstructed_data[0].real
    data_prediction[-pointsperday:] = data_prediction[-pointsperday:] + error_forecast
    # ----some special correction on the decomposition-----
    zeroindex = np.where(dmddataset_org[-1] <= 0.01)[0]
    data_prediction[days * pointsperday + zeroindex] = dmddataset_org[-1][zeroindex]
    # -----special correction is done--------------
    #data_predicion_org = data_prediction[:]
    plt.plot(cut + hodmd.dmd_timesteps - days * pointsperday, data_practical, '-', label='the practical signal',
             color='g')
    plt.plot(cut + hodmd.dmd_timesteps - days * pointsperday, data_prediction, '--', label='DMD output', color='r')
    plt.vlines(cut, 0, 20, colors="black", linestyles="--")
    plt.title(cut)
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(pointsperday), data_practical[-pointsperday:], '-', label='the practical signal', color='g')
    plt.plot(np.arange(pointsperday), data_predicion_org[-pointsperday:], '--', label='forecast(without error forecast)', color='r')
    plt.grid()
    plt.xticks(np.arange(0, 300, 10))
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(pointsperday), data_practical[-pointsperday:], '-', label='the practical signal', color='g')
    plt.plot(np.arange(pointsperday), data_prediction[-pointsperday:], '--', label='forecast', color='r')
    plt.xticks(np.arange(0, 300, 10))
    plt.grid()
    plt.legend()
    plt.show()




    print(colored('------Error evaluation for with DMD and error prediction:------', 'green'))
    data_prediction_eva = data_prediction[-pointsperday:]
    data_practical_eva = data_practical[-pointsperday:]
    ev_result = ev(data_prediction_eva, data_practical_eva)
    for key in ev_result:
        print '%s: %s' % (key, ev_result[key])

    print(colored('------Error evaluation for without DMD:------', 'green'))
    data_prediction_eva = dmddataset_org[-1]+error_forecast
    ev_result = ev(data_prediction_eva, data_practical_eva)
    for key in ev_result:
        print '%s: %s' % (key, ev_result[key])
#test = dd(maresult[1], days, )


# do some gpr stuff here:









