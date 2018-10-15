import numpy as np
import matplotlib.pyplot as plt
from readdata import read as rd
from pydmd import HODMD
def fuc(x):
    fucresult = np.cos(x)*np.sin(np.cos(x))+np.cos(x*.2)*np.sin(x)+x*(25-x)/100
    return fucresult
import plotcheck
#just test
#path = "/home/bwei/PycharmProjects/data lib/long wind"
#windset = rd(path)
#name = raw_input('the name of data set?')
x = np.linspace(0, 30, 128)
realwindset = fuc(x)
realwindset.shape = (len(realwindset),)
realwindset = realwindset + np.random.rand(128)
#x = np.linspace(1, len(realwindset), len(realwindset))
hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=58).fit(realwindset)
hodmd.reconstructed_data.shape
hodmd.plot_eigs()
hodmd.original_time['dt'] = hodmd.dmd_time['dt'] = x[1] - x[0]
hodmd.original_time['t0'] = hodmd.dmd_time['t0'] = x[0]
hodmd.original_time['tend'] = hodmd.dmd_time['tend'] = x[-1]

plt.plot(hodmd.original_timesteps, realwindset, '.', label='realwindset')
plt.plot(x, realwindset, '-', label='original function')
plt.plot(hodmd.dmd_timesteps, hodmd.reconstructed_data[0].real, '--', label='DMD output')
plt.legend()
plt.show()

extrap = 100
extrapx = np.linspace(0, extrap, extrap*20)
original_fuc = fuc(extrapx)

hodmd.dmd_time['tend'] = extrap
fig = plt.figure(figsize=(15, 5))
plt.plot(hodmd.original_timesteps, realwindset, '.', label='snapshots')
plt.plot(extrapx, original_fuc, '-', label='original function')
plt.plot(hodmd.dmd_timesteps, hodmd.reconstructed_data[0].real, '--', label='DMD output')
plt.legend()
plt.show()
reconstruct = np.dot(hodmd.modes, hodmd.dynamics)
plotcheck.pl(np.transpose(reconstruct))
