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
    forecast_result = []
    for loop in np.arange(len(differsets)):
        imfs = emd(differsets[loop]) # do the EMD
        nimfs = len(imfs)
        extrema_upper_index_vector = []# record, make no sense
        extrema_lower_index_vector = []
        forecast_value_vector = []
        for n in np.arange(nimfs): # try to figure out the trend and give prediction
            #---------wash the extremas----------------------------------------------------
            extrema_upper_index = extrema(imfs[n], np.greater_equal)[0] # max extrema
            neighbours = []
            for i in np.arange(len(extrema_upper_index)-1): # clean the indexes which close to each other
                if extrema_upper_index[i]-extrema_upper_index[i+1] == -1:
                    neighbours.append(i)
            extrema_upper_index = np.delete(extrema_upper_index, neighbours)
            extrema_upper_index = np.delete(extrema_upper_index, np.where((extrema_upper_index == 0) |
                                                                          (extrema_upper_index == len(imfs[n])-1)))
            neighbours = []

            extrema_lower_index = extrema(imfs[n], np.less_equal)[0]# min exrema
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
                    #--some basic calculations --------#
            extrema_upper_value = imfs[n][extrema_upper_index]
            extrema_lower_value = imfs[n][extrema_lower_index]
            extremas = np.unique(np.hstack([extrema_upper_index, extrema_lower_index]))
            if extremas.any():
                last_extrema = extremas[-1]
            else:
                last_extrema = len(imfs[n])-1

            if len(extrema_upper_index)+len(extrema_lower_index) <= 0: # if there is no real extrema
                distance = last_extrema #means that there is no enough extremas to do the calculation
                amplitude_upper_ema = max(imfs[n])
                amplitude_lower_ema = min(imfs[n])
                step = abs(amplitude_upper_ema-amplitude_lower_ema)/distance
                reference_amplitude = abs(imfs[n][-1])+2*step
            elif len(extrema_upper_index)+len(extrema_lower_index) == 1:# if there is only one extrema
                distance = len(imfs[n])-last_extrema
                amplitude_upper_ema = max(imfs[n][last_extrema], imfs[n][-1])
                amplitude_lower_ema = min(imfs[n][last_extrema], imfs[n][-1])
                step = abs(amplitude_upper_ema-amplitude_lower_ema)/distance
                reference_amplitude = abs(imfs[n][-1]) + 2 * step
            else:
                amplitude_upper_ema = ema(extrema_upper_value, alpha=0.6)
                amplitude_lower_ema = ema(extrema_lower_value, alpha=0.6)
                nextremas = min(len(extrema_lower_index), len(extrema_upper_index))
                distance_set = abs(extrema_upper_index[-nextremas:]-extrema_lower_index[-nextremas:])
                distance = ema(distance_set, alpha=0.6)
                step = abs(amplitude_upper_ema - amplitude_lower_ema) / distance
                reference_amplitude = abs(amplitude_lower_ema) * 0.25 + abs(amplitude_upper_ema) * 0.25 + abs(
                    imfs[n][last_extrema]) * 0.5
        # do the rough forecast from here:
            if n >= np.floor(nimfs/2):
                step = abs(imfs[n][-1]-imfs[n][-2])

            if imfs[n][last_extrema]*imfs[n][-1] < 0:# if the last point has already crossed the axis
                if distance <= 1.58: # have to switch the direction
                    forecast_value = imfs[n][-1]-imfs[n][-1]/abs(imfs[n][-1])*step
                else:
                    if abs(imfs[n][-1])+step > reference_amplitude:
                        forecast_value = imfs[n][-1]/abs(imfs[n][-1])*reference_amplitude
                    else:
                        forecast_value = imfs[n][-1]/abs(imfs[n][-1])*(abs(imfs[n][-1])+step)
            elif imfs[n][-1]-imfs[n][-2] == 0:
                forecast_value = 0 #give up the forecast
            else:
                if distance < 1.1: #means have a more often switch
                    forecast_value = imfs[n][-1] - step * (imfs[n][-1] - imfs[n][-2]) / abs((imfs[n][-1] - imfs[n][-2])) #change the direction
                else:
                    forecast_value = imfs[n][-1]+step*(imfs[n][-1]-imfs[n][-2])/abs((imfs[n][-1]-imfs[n][-2])) # continue with the trend

            forecast_value_vector.append(forecast_value)
        forecast_result.append((sum(forecast_value_vector)))


            #-------------------------the derivation is done------------------------------




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
            plt.scatter(x[-1]+1, forecast_value_vector[loop-1], marker='o', c='black', s=50)
            plt.hlines(0, 0, len(differsets[0]), colors="black", linestyles="--")
            plt.title(loop)
        plt.show()

    return forecast_result

def ema(data, alpha): #simple function to give a ema as u want
        emaresult = data[0]
        for n in np.arange(len(data)-1):
            emaresult = emaresult*(1-alpha)+alpha*data[n+1]
        return emaresult





