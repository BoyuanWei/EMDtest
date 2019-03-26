# the prediction by dmd with removing the zero zones
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

path = "/home/bwei/PycharmProjects/data lib/PVdata5min6month.csv"
#windset = rd(path)
#name = raw_input('the name of data set?')
#realwindset = windset[name]
pointsperday = 288 # CHANGE HERE FOR DIFFERENT RESOLUTION
realwindset = readcsv.rd(path)
#realwindset=hour(realwindset)#############
windsetoriginal= realwindset
realwindset.shape = (len(realwindset),)
#--------------------swithes---------------
moving_average_flag = 0  # ------------------CHANGE WHETHER THE MOVING AVERAGE IS NEEDED HERE.--------------
zero_zone_cut = 1 # ------------------CHANGE WHETHER THE ZERO ZONE CUT IS NEEDED HERE.--------------


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

def non2full(dmd_prediction, zero_data_index, pointsperday):
    full_prediction_one_day = np.array([0.00]*pointsperday) # restore the nonzero prediction result to a full day
    position_flag = 0
    for n in np.arange(pointsperday):
        if n in zero_data_index:
            pass
        else:
            full_prediction_one_day[n] = dmd_prediction[position_flag]
            position_flag = position_flag+1
    return full_prediction_one_day

days = input('how many days will be generated for forecasting?')
cut = input('from which day the rest will be testing set?<'+str(len(realwindset)/pointsperday))

cut = cut*pointsperday
dmddataset = realwindset[cut-days*pointsperday:cut]

#------- moving average process------------------
if moving_average_flag == 1:
    day_ahead = 10
    data_for_ma = realwindset[cut-days*pointsperday*day_ahead:cut]
    moving_average_set = ma(data_for_ma, days, pointsperday)[0]
    dmddataset = []
    for n in np.arange(len(moving_average_set)):
        dmddataset.extend(moving_average_set[n])
    dmddataset = np.array(dmddataset)

#--------clean the zero zone---------
if zero_zone_cut == 1:
    dmddataset_ma = ma(dmddataset, days, pointsperday)[0][-1]
    zero_data_index = np.where(dmddataset_ma == 0)
    zero_data_index_org = zero_data_index
    for n in np.arange(days-1):
        zero_data_index = np.append(zero_data_index, zero_data_index+(n+1)*pointsperday)
    dmddataset = np.delete(dmddataset, zero_data_index)

#-----core prediction---------------#
hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=len(dmddataset)/days).fit(dmddataset)
hodmd.reconstructed_data.shape
hodmd.plot_eigs()
hodmd.dmd_time['tend'] = len(dmddataset)/days*(days+1)-1# since it starts from zero
dmd_output = hodmd.reconstructed_data[0].real
dmd_prediction = dmd_output[-len(dmddataset)/days:]

if zero_zone_cut == 1 :
    full_prediction_one_day = non2full(dmd_prediction, zero_data_index_org[0], pointsperday)
else:
    full_prediction_one_day = dmd_prediction

#----one pic about the with DMD output
fig = plt.figure(figsize=(20, 10))
plt.plot(cut-len(dmddataset)+np.arange(len(dmddataset)+pointsperday),
         np.append(dmddataset, realwindset[cut:cut+pointsperday]), '-', label='practical')
data_practical = realwindset[cut:cut+pointsperday]
plt.plot(cut-len(dmddataset)+np.arange(len(dmddataset)+pointsperday),
         np.append(dmd_output[:-len(dmd_output)/(days+1)], full_prediction_one_day), '--', label='DMD output', color='r')
plt.vlines(cut, 0, 20, colors="black", linestyles="--")
plt.legend()
plt.show()



#Start the error evaluation section from here(:
print(colored('------Error evaluation for with DMD:------', 'green'))
data_prediction_dmd = full_prediction_one_day
data_practical_eva = realwindset[cut:(cut+pointsperday)]
ev_result = ev(data_prediction_dmd, data_practical_eva)
for key in ev_result:
    print '%s: %s' % (key, ev_result[key])











