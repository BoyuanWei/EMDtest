import numpy as np
import matplotlib.pyplot as plt
from readdata import read as rd
from pydmd import HODMD
from termcolor import *
path = "/home/bwei/PycharmProjects/EMDtest/long wind"
windset = rd(path)
name = raw_input('the name of data set?')
realwindset = windset[name]
realwindset.shape = (len(realwindset),)
x = np.linspace(1, len(realwindset), len(realwindset))
hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=400).fit(realwindset)
hodmd.reconstructed_data.shape
hodmd.plot_eigs()
hodmd.original_time['dt'] = hodmd.dmd_time['dt'] = x[1] - x[0]
hodmd.original_time['t0'] = hodmd.dmd_time['t0'] = x[0]
hodmd.original_time['tend'] = hodmd.dmd_time['tend'] = x[-1]

plt.plot(hodmd.original_timesteps, realwindset, '.', label='realwindset')
plt.plot(hodmd.original_timesteps, realwindset, '-', label='original function')
plt.plot(hodmd.dmd_timesteps, hodmd.reconstructed_data[0].real, '--', label='DMD output')
plt.legend()
plt.show()

hodmd.dmd_time['tend'] = 5000
fig = plt.figure(figsize=(15, 5))
plt.plot(hodmd.original_timesteps, realwindset, '.', label='snapshots')
plt.plot(x, realwindset, '-', label='original function')
plt.plot(hodmd.dmd_timesteps, hodmd.reconstructed_data[0].real, '--', label='DMD output')
plt.legend()
plt.show()
