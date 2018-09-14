import numpy as np
import matplotlib.pyplot as plt
from readdata import read as rd
from PyEMD import EMD
from termcolor import *
from pydmd import HODMD
path = "/home/bwei/PycharmProjects/data lib/long wind"# the folder path of data
windset = rd(path)
name = []
while name != 00000:
    name = raw_input('the name of data set? form like s?day_? ?, s from 1-24, day from 1-7')
    if name == 'exit': # the exit command
        break
    if name == 'composite': # to composite the imfs after a EMD
        if 'imfs' in locals():
            composite = raw_input('which orders? or range(use lick "f,a,b". or use positive orders which starts from 0')
            composite = [n for n in composite.split(',')]
            if 'f' in composite:
                composite = np.arange(int(composite[1])-1, int(composite[2]), 1)
            else:
                composite = np.array(map(int, composite))
            newsignal = np.sum(imfs[composite], axis=0)
            plt.figure()
            plt.plot(x, newsignal)
            plt.show()
        else:
            print(colored("You have to do at least one EMD first.", 'red'))
        continue
    if name == 'dmd': # do a DMD after a composition among IMFs
        if 'newsignal' in locals():
            d = raw_input('what is the d in DMD-d?')
            x = np.linspace(1, len(newsignal), len(newsignal))
            hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=int(d)).fit(newsignal)
            hodmd.reconstructed_data.shape
            hodmd.plot_eigs()
            hodmd.original_time['dt'] = hodmd.dmd_time['dt'] = x[1] - x[0]
            hodmd.original_time['t0'] = hodmd.dmd_time['t0'] = x[0]
            hodmd.original_time['tend'] = hodmd.dmd_time['tend'] = x[-1]
            #draw


            plt.plot(hodmd.dmd_timesteps, hodmd.reconstructed_data[0].real, '--', label='DMD output')
            plt.legend()
            plt.show()
            farseer = raw_input('want to see a forecast?(y/n)')
            if farseer == 'y':
                tendlength = raw_input('then, for how long?')
                hodmd.dmd_time['tend'] = int(tendlength)
                fig = plt.figure(figsize=(15, 5))
                plt.plot(hodmd.original_timesteps, newsignal, '.', label='snapshots')
                if 'windsetoriginal' in locals():
                    y = np.linspace(1, len(windsetoriginal), len(windsetoriginal))
                    plt.plot(y, windsetoriginal, '-', label='the practical signal')
                plt.plot(hodmd.dmd_timesteps, hodmd.reconstructed_data[0].real, '--', label='DMD output')
                plt.legend()
                plt.show()
            else:
                continue
        else:
            print(colored("You have to do at least one composition first.", 'red'))
        continue
    if name == 'joint': # to make a joint signal, like combine the first part of A with the second part of B, then EMD
        joint = raw_input('Which two? From where?')
        joint = [n for n in joint.split(',')]
        joint[-1] = int(joint[-1])
        realwindset = np.append(windset[joint[0]][:joint[-1]+1], windset[joint[1]][joint[-1]:-1])
        x = np.linspace(1, len(realwindset), len(realwindset))
        emd = EMD()
        realwindset.shape = (len(realwindset),)
        imfs = emd(realwindset)
        size = imfs.shape
        plt.figure()
        plt.plot(x, realwindset)
        plt.title(joint)
        plt.show()
        plt.figure(figsize=(20, 18))
        for loop in range(1, size[0] + 1):
            plt.subplot(size[0], 1, loop)
            plt.plot(x, imfs[loop - 1])
            plt.title(loop)
        plt.show()
        continue
    if name not in windset.keys():# in case the user give some wrong names
        print(colored('This dataset is not exist', 'red'))
        continue
    cut = raw_input('Cut the zeros head and end?[y/n] or from somewhere?[from]') # an option to cut the zero zones in the dataset
    if cut == 'y':
        cutindex = [np.nonzero(windset[name])[0][0], np.nonzero(windset[name])[0][-1]]
        realwindset = windset[name][cutindex[0]:cutindex[1]+1]
    elif cut == 'from':
        cutfrom = raw_input('from where?')
        windsetoriginal = windset[name]
        realwindset = windset[name][0:int(cutfrom)]
    else:
        realwindset = windset[name]
    x = np.linspace(1, len(realwindset), len(realwindset))
    emd = EMD()
    realwindset.shape = (len(realwindset),)
    imfs = emd(realwindset)
    size = imfs.shape
#-----------Ploting starts from here-----------
    plt.figure()
    plt.plot(x, realwindset)
    plt.title(name)
    plt.show()
    plt.figure(figsize=(20, 18))
    for loop in range(1, size[0] + 1):
        plt.subplot(size[0], 1, loop)
        plt.plot(x, imfs[loop - 1])
        plt.title(loop)
    plt.show()
