import numpy as np
import matplotlib.pyplot as plt
from readdata import read as rd
from pyhht.emd import EMD
from termcolor import *
path = "/home/bwei/PycharmProjects/EMDtest/long wind"
windset = rd(path)
name = []
while name != 00000:
    name = raw_input('the name of data set? form like s?day_? ?, s from 1-24, day from 1-7')
    if name == 'exit':
        break
    if name == 'composite':
        if 'imfs' in locals():
            composite = raw_input('which orders?')
            composite = [int(n) for n in composite.split(',')]
            newsignal = np.sum(imfs[composite], axis=0)
            plt.figure()
            plt.plot(x, newsignal)
            plt.show()
        else:
            print(colored("You have to do at least one EMD first.", 'red'))
        continue
    if name not in windset.keys():
        print(colored('This dataset is not exist', 'red'))
        continue
    cut = raw_input('Cut the zeros head and end?[y/n]')
    if cut == 'y':
        cutindex = [np.nonzero(windset[name])[0][0], np.nonzero(windset[name])[0][-1]]
        realwindset = windset[name][cutindex[0]:cutindex[1]+1]
    else:
        realwindset = windset[name]
    x = np.linspace(1, len(realwindset), len(realwindset))
    decomposer = EMD(realwindset)
    imfs = decomposer.decompose()
    size = imfs.shape
    plt.figure()
    plt.plot(x, realwindset)
    plt.title(name)
    plt.show()
    plt.figure(figsize=(20, 18))
    for loop in range(1, size[0]+1):
        plt.subplot(size[0], 1, loop)
        plt.plot(x, imfs[loop-1])
        plt.title(loop)
    plt.show()




