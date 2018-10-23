# a pak to give the evaluation to the regression
# also a pak to give "prediction" on the random parts
import numpy as np
from math import sqrt
from PyEMD import EMD
import matplotlib.pyplot as plt

def ev(data_prediction_eva, data_practical_eva):
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

def randomextract(data_origin, data_ema, points_per_day=288):
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

def doemd(data):
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


