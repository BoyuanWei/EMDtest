import numpy as np
import readcsv
import matplotlib.pyplot as plt
from plotcheck import pl

#read 2016:
path_2016= "/home/bwei/PycharmProjects/data lib/pv_2016.csv"
dataset_2016 = readcsv.rd(path_2016)
dataset_2016[np.where(dataset_2016>50)]=0
#read_2017:
path_2017= "/home/bwei/PycharmProjects/data lib/pv_2017.csv"
dataset_2017 = readcsv.rd(path_2017)
dataset_2017[np.where(dataset_2017>50)]=0
#read_2018:

path_2018= "/home/bwei/PycharmProjects/data lib/pv_2018.csv"
dataset_2018 = readcsv.rd(path_2018)

points_per_day = 288
rounds = len(dataset_2016)/points_per_day

# sum up func:
def sumup(data, rounds, point_per_day=288):
    sum_vector = []
    for n in np.arange(rounds):
        sum_vector_one = np.sum(data[n*point_per_day:(n+1)*point_per_day])
        sum_vector.append(sum_vector_one)
    return np.array(sum_vector)

#sum up 2016-2018:
sum_2016 = sumup(dataset_2016,rounds,points_per_day)
sum_2017 = sumup(dataset_2017,rounds,points_per_day)
sum_2018 = sumup(dataset_2018,rounds,points_per_day)

differ_1 = sum_2017-sum_2016
differ_2 = sum_2018-sum_2017
differ_3 = sum_2018-sum_2016

sections = 5 # see how many sections

#devide the sections
range_sections = sum_2016.max()/sections

pl(differ_1)
pl(differ_2)
pl(differ_3)

field_2016 = np.floor(sum_2016/range_sections)
field_2017 = np.floor(sum_2017/range_sections)
field_2018 = np.floor(sum_2018/range_sections)

field_differ_1 = np.abs(field_2016-field_2017)
field_differ_2 = np.abs(field_2017-field_2018)
field_differ_3 = np.abs(field_2016-field_2018)

relevance_ratio_1 = len(np.where(field_differ_1<=1)[0])/1.00/len(field_differ_1)
print ('The relevance ratio between 2016 and 2017 is'), relevance_ratio_1

relevance_ratio_2 = len(np.where(field_differ_2<=1)[0])/1.00/len(field_differ_2)
print ('The relevance ratio between 2017 and 2018 is'), relevance_ratio_2

relevance_ratio_3 = len(np.where(field_differ_3<=1)[0])/1.00/len(field_differ_3)
print ('The relevance ratio between 2016 and 2018 is'), relevance_ratio_3
