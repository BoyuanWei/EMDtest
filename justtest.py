import numpy as np
import readcsv
import plotcheck as pc
aaa = np.array([1,2,3,4,5,6])
bbb = np.array([2,3,4,5,6,7])
ccc = np.array([3,4,5,6,7,8])

imfs = np.array([])
kkkk = np.array([aaa, bbb, ccc])

path = '/home/bwei/PycharmProjects/data lib/pvtotal.csv'
data = readcsv.rd(path)
