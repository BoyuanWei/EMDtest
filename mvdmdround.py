# moving average on historical data and then do the dmd, hope it works
# This is the iteration version of mvdmd, to do some leg labour works
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

path = "/home/bwei/PycharmProjects/data lib/pvtotal.csv"
#windset = rd(path)
#name = raw_input('the name of data set?')
#realwindset = windset[name]
pointsperday = 288 # CHANGE HERE FOR DIFFERENT RESOLUTION
realwindset = readcsv.rd(path)
#realwindset=hour(realwindset)#############
windsetoriginal= realwindset
realwindset.shape = (len(realwindset),)
#data reading is done
succ=[]
action = []
ranges = np.arange(9000, 26100)
for iters in ranges:
    def ma(data, days_to_keep, points_per_day, alpha=0.2):
        days_covered = int(np.floor(len(data)/points_per_day))
        points_covered = days_covered*points_per_day
        daysdata = []
        onedaydata = data[len(data)-points_covered:len(data)-points_covered+points_per_day]
        for loop in np.arange(days_covered-1):
            onedaydata = onedaydata*(1-alpha)+alpha*data[len(data)-points_covered+points_per_day*(loop+1):
                                                         len(data)-points_covered+points_per_day*(loop+1)+points_per_day]
            daysdata.append(onedaydata)
        return daysdata[-days_to_keep:], daysdata

#moving average definition is done
    days = 5
    cut = iters
    madataset = realwindset[:cut]
    maresult = ma(madataset, days, pointsperday)
    dmddataset_org = maresult[0]
    dmddataset = []
    for n in np.arange(len(dmddataset_org)):
        dmddataset.extend(dmddataset_org[n])
    dmddataset = np.array(dmddataset)
    hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=pointsperday).fit(dmddataset)
    hodmd.reconstructed_data.shape
    hodmd.dmd_time['tend'] = (days+1)*pointsperday-1# since it starts from zero

#----one pic about the with DMD one
#fig = plt.figure(figsize=(20, 10))
#plt.plot(hodmd.original_timesteps+cut-days*pointsperday, dmddataset, '.', label='snapshots')
    data_practical = windsetoriginal[cut+hodmd.dmd_timesteps-days*pointsperday]
    data_prediction = hodmd.reconstructed_data[0].real
#----some special correction on the decomposition-----
    zeroindex = np.where(dmddataset_org[-1] <= 0.01)[0]
    data_prediction[days*pointsperday+zeroindex] = dmddataset_org[-1][zeroindex]


    dmd_difference_set = dd(maresult[1], days)
    differset = re(madataset, dmd_difference_set) # calculate the differences between ema results and practical data
    differset = datawash(differset)
    bb = baynetwork(differset, dmddataset_org[-1])
    error_forecast = np.array(bino(bb, differset, dmddataset_org[-1]))
    realerror = data_practical[-pointsperday:]-data_prediction[-pointsperday:]
    succ_rate = percents(error_forecast, realerror, dmddataset_org[-1])
    succ.append(succ_rate[0])
    action.append(succ_rate[1])

fig = plt.figure(figsize=(20, 10))
plt.plot(ranges, succ, '.', label='successful rate')
#plt.plot(cut+hodmd.dmd_timesteps-days*pointsperday, data_practical, '-', label='the practical signal', color='g')
plt.plot(ranges, action, '--', label='action rate', color='r')
plt.legend()
plt.show()



# do some gpr stuff here:









