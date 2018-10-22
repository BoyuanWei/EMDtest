#this is a toy to see whether the data of certain point can really be regressed successfully
import numpy as np
import readcsv
from evaluation import ev as ev
from plotcheck import pl as pl
from scipy.fftpack import fft, ifft
import seaborn
import matplotlib.pyplot as plt


def ma(data, days_to_keep, alpha=0.2, points_per_day=288):
    days_covered = int(np.floor(len(data) / points_per_day))
    points_covered = days_covered * points_per_day
    daysdata = []
    onedaydata = data[len(data) - points_covered:len(data) - points_covered + points_per_day]
    for loop in np.arange(days_covered - 1):
        onedaydata = onedaydata * (1 - alpha) + alpha * data[len(data) - points_covered + points_per_day * (loop + 1):
                                                             len(data) - points_covered + points_per_day * (
                                                                         loop + 1) + points_per_day]
        daysdata.append(onedaydata)
    return daysdata[-days_to_keep:]

path = "/home/bwei/PycharmProjects/data lib/pvtotal.csv"
#windset = rd(path)
#name = raw_input('the name of data set?')
#realwindset = windset[name]
pointsperday = 288 # CHANGE HERE FOR DIFFERENT RESOLUTION
realwindset = readcsv.rd(path)
windsetoriginal= realwindset
realwindset.shape = (len(realwindset),)
point_index = 236#should < 288
days_to_keep = 90
mvdata = ma(realwindset, days_to_keep)
mvset = []
for n in np.arange(len(mvdata)):
    mvset.extend(mvdata[n])
practicaldata = realwindset[-len(mvset):]
differset = practicaldata-mvset
newplotdata = []
for n in np.arange(days_to_keep):
    newplotdata.append(differset[point_index+n*288])
pl(np.array(newplotdata))

x = np.linspace(0, 1, days_to_keep)
yy = fft(np.array(newplotdata))
yreal = yy.real
yimag = yy.imag
yf = abs(yy)
yf1 = abs(yy)/len(x)
yf2 = yf1[range(int(len(x)/2))]

xf = np.arange(len(np.array(newplotdata)))
xf1 = xf
xf2 = xf[range(int(len(x)/2))]


plt.subplot(221)
plt.plot(x[0:50],np.array(newplotdata)[0:50])
plt.title('Original wave')

plt.subplot(222)
plt.plot(xf,yf,'r')
plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')

plt.subplot(223)
plt.plot(xf1,yf1,'g')
plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')

plt.subplot(224)
plt.plot(xf2,yf2,'b')
plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')

plt.show()