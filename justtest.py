import numpy as np
import readcsv
from plotcheck import pl
import GPy
from matplotlib import pyplot as plt
from pydmd import HODMD
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


path = "/home/bwei/PycharmProjects/data lib/PVhourly6months.csv"
realwindset = readcsv.rd(path,datawashflag=1)

pointsperday=24
days = 2

hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=pointsperday).fit(realwindset[-48:])
hodmd.reconstructed_data.shape
hodmd.plot_eigs()
hodmd.dmd_time['tend'] = (days+1)*pointsperday-1# since it starts from zero