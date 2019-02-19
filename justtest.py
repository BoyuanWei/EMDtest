import numpy as np
import readcsv
import plotcheck as pc
import GPy
from matplotlib import pyplot as plt
aaa = np.array([1,2,3,4,5,6,8,19,30])
bbb = np.array([2,10,4,5,12,7])
ccc = np.array([3,4,5,6,7,8])
imfs = np.array([])
kkkk = np.array([aaa, bbb, ccc])


Percentile = np.percentile(aaa,[0,25,50,75,100])
IQR = Percentile[3] - Percentile[1]
UpLimit = Percentile[3]+IQR*1.5
DownLimit = Percentile[1]-IQR*1.5

aaa[np.where(aaa>UpLimit)]=UpLimit

avvv=[{}]*3
