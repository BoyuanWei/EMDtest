# a pak to give the evaluation to the regression
# also a pak to give "prediction" on the random parts
import numpy as np
from math import sqrt
from PyEMD import EMD
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema as extrema

def ev(data_prediction_eva, data_practical_eva): #see the accuracy
    evaluations = {}
    mean_data_practical_eva = np.mean(data_practical_eva)
    error = []
    absPercentError = []
    for i in range(len(data_practical_eva)):
        error.append(data_practical_eva[i] - data_prediction_eva[i])
        if data_practical_eva[i] == 0:
            absPercentError.append(abs(data_practical_eva[i] - data_prediction_eva[i]) / abs(mean_data_practical_eva))
        else:
            absPercentError.append(abs(data_practical_eva[i] - data_prediction_eva[i]) / abs(data_practical_eva[i]))
    # print("Errors: ", error)
    # print(error)
    # ------squared error------
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))
    # print("Square Error: ", squaredError)
    # print("Absolute Value of Error: ", absError)
    # ------- MSE---------
    evaluations['MSE'] = sum(squaredError) / len(squaredError)

    # -----RMSE----------
    evaluations['RMSE'] = sqrt(sum(squaredError) / len(squaredError))
    # - ------   MAE - -  - -- -- - -
    evaluations['MAE'] = sum(absError) / len(absError)

    # - ------ MAPE-----------

    evaluations['MAPE(quasi)'] = sum(absPercentError) / len(absPercentError)

    #

    targetDeviation = []
    targetMean = sum(data_practical_eva) / len(data_practical_eva)
    for val in data_prediction_eva:
        targetDeviation.append((val - targetMean) * (val - targetMean))
    evaluations['Practical data Variance'] = sum(targetDeviation) / len(targetDeviation)
    evaluations['Practical data Standard Deviation'] = sqrt(sum(targetDeviation) / len(targetDeviation))
    return evaluations

def randomextract(data_origin, data_ema, points_per_day=288):# generate the differsets of everypoint in 288 (oneday)
    days = len(data_ema)
    differ = []
    differset = []
    usefuldata = data_origin[-points_per_day*days:]
    for loop in np.arange(points_per_day):
        for n in np.arange(days):
            differ.append(usefuldata[loop+n*points_per_day]-data_ema[n][loop])
        differset.append(np.array(differ))
        differ = []
    return differset

def drawemd(data): # draw the pictures of the emd.
    emd = EMD()
    imfs = emd(data)
    size = imfs.shape
    x = np.linspace(1, len(data), len(data))
    plt.figure()
    plt.plot(x, data, marker='.', markerfacecolor='blue', markersize=10)
    plt.show()
    plt.figure(figsize=(20, 18))
    for loop in range(1, size[0] + 1):
        plt.subplot(size[0], 1, loop)
        plt.plot(x, imfs[loop - 1], marker='.', markerfacecolor='blue', markersize=10)
        plt.hlines(0, 0, len(data), colors="black", linestyles="--")
        plt.title(loop)
    plt.show()


def pointprediction(differsets, draw=0): # forecast the next differ of members in differsets
    emd = EMD()
    for loop in np.arange(len(differsets)):
        imfs = emd(differsets[loop]) # do the EMD
        nimfs = len(imfs)
        extrema_upper_index_vector = []# record, make no sense
        extrema_lower_index_vector = []
        for n in np.arange(nimfs): # try to figure out the trend and give prediction
            #---------wash the extremas----------------------------------------------------
            extrema_upper_index = extrema(imfs[n], np.greater_equal) # max extrema
            neighbours = []
            for i in np.arange(len(extrema_upper_index)-1): # clean the indexes which close to each other
                if extrema_upper_index[i]-extrema_upper_index[i+1] == -1:
                    neighbours.append(i)
            extrema_upper_index = np.delete(extrema_upper_index, neighbours)
            extrema_upper_index = np.delete(extrema_upper_index, np.where((extrema_upper_index == 0) |
                                                                          (extrema_upper_index == len(imfs[n])-1)))
            neighbours = []

            extrema_lower_index = extrema(imfs[n], np.less_equal)# min exrema
            for i in np.arange(len(extrema_lower_index)-1): # clean the indexes which close to each other
                if extrema_lower_index[i]-extrema_lower_index[i+1] == -1:
                    neighbours.append(i)
            extrema_lower_index = np.delete(extrema_lower_index, neighbours)
            extrema_lower_index = np.delete(extrema_lower_index, np.where((extrema_lower_index == 0) |
                                                                          (extrema_lower_index == len(imfs[n]-1)-1)))
            if draw == 1:
                extrema_upper_index_vector.append(extrema_upper_index)
                extrema_lower_index_vector.append(extrema_lower_index)

            #------------------------ the derivation starts from here---------------------


            #-------------------------the derivation is done------------------------------




        #---------------wash done-----------------------------------------------
            extrema_upper_value = imfs[n][extrema_upper_index]
            extrema_lower_value = imfs[n][extrema_lower_index]




    # a drawing to show some result for bugging
    if draw == 1:
        size = imfs.shape
        x = np.arange(len(differsets[0]))
        plt.figure()
        plt.plot(x, differsets[0], marker='.', markerfacecolor='blue', markersize=6)
        plt.show()
        plt.figure(figsize=(20, 18))
        for loop in range(1, size[0] + 1):
            plt.subplot(size[0], 1, loop)
            plt.plot(x, imfs[loop - 1], marker='.', markerfacecolor='blue', markersize=6)
            plt.scatter(extrema_upper_index_vector[loop-1], imfs[loop-1][extrema_upper_index_vector[loop-1]], c='red',
                        marker='+', s=50)
            plt.scatter(extrema_lower_index_vector[loop-1], imfs[loop-1][extrema_lower_index_vector[loop-1]], marker='+',
                        color='green', s=50)
            plt.hlines(0, 0, len(differsets[0]), colors="black", linestyles="--")
            plt.title(loop)
        plt.show()

    return extrema_upper_index_vector, extrema_lower_index_vector





