import numpy as np
import matplotlib.pyplot as plt
from readdata import read as rd
path = "/home/bwei/PycharmProjects/EMDtest/data"
windset=rd(path)
x = np.linspace(1, 288, 288)
name=[]
while name != 00000:
   name = raw_input('the name of data set? form like s?day_? ?, s from 1-24, day from 1-7')
   if name == 'exit':
       break
   plt.figure()
   plt.plot(x, windset[name])
   plt.show()
