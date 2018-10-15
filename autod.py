import numpy as np
import matplotlib.pyplot as plt
from readdata import read as rd
from termcolor import *
from pydmd import HODMD

path = "/home/bwei/PycharmProjects/data lib/long wind"# the folder path of data
windset = rd(path)
name = []
errorvector = []
#name = raw_input('the name of data set? form like s?day_? ?, s from 1-24, day from 1-7')
#some new toys:

def fuc(x):
    fucresult = np.cos(x)*np.sin(np.cos(x))+np.cos(x*.2)*np.sin(x)+np.cos(x/np.cos(x)+x/5)
    return fucresult
x = np.linspace(0, 10, 128)
realwindset = fuc(x)
# toy end

#realwindset = windset[name]
rounds = raw_input('from ?')
rounds = int(rounds)
roundt = raw_input('to?')
roundt = int(roundt)
drange = np.arange(rounds, roundt+1, 1)
realwindset.shape = (len(realwindset),)
x = np.linspace(1, len(realwindset), len(realwindset))

#new:

for loop in drange:
    hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=int(loop)).fit(realwindset) # key functional sentence
#hodmd.reconstructed_data.shape
    errors = realwindset-hodmd.reconstructed_data[0]
    sumerror = np.sum(abs(errors))
    errorvector.append(sumerror)
y = np.arange(1, len(errorvector)+1, 1)
plt.plot(y, errorvector, '--', label='errors')
#plt.plot(hodmd.dmd_timesteps, realwindset, '-', label='realwindset')
#plt.plot(hodmd.dmd_timesteps, errors, '.', label='errors')
plt.legend()
plt.show()
