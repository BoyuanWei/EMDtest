# a pak to give the evaluation to the regression
# also a pak to give "prediction" on the random parts
import numpy as np
from math import sqrt
from PyEMD import EMD
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema as extrema
from pydmd import HODMD
import GPy
from pyramid.arima import auto_arima
from IPython.display import display

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
            if len(extrema_upper_index) + len(extrema_lower_index) <= 0:  # if there is no real extrema
                distance = last_extrema  # means that there is no enough extremas to do the calculation
                amplitude_upper_ema = max(imfs[n])
                amplitude_lower_ema = min(imfs[n])
                step = abs(amplitude_upper_ema - amplitude_lower_ema) / distance
                forecast_value = imfs[n][-1] + step * (imfs[n][-1] - imfs[n][-2]) / abs((imfs[n][-1] - imfs[n][-2])) # just extend the tread
            elif len(extrema_upper_index) + len(extrema_lower_index) == 1:  # if there is only one extrema
                distance = len(imfs[n]) - last_extrema
                amplitude_upper_ema = max(imfs[n][last_extrema], imfs[n][-1])
                amplitude_lower_ema = min(imfs[n][last_extrema], imfs[n][-1])
                step = abs(amplitude_upper_ema - amplitude_lower_ema) / distance
                #reference_amplitude = abs(imfs[n][-1]) + 2 * step
                forecast_value = imfs[n][-1] + step * (imfs[n][-1] - imfs[n][-2]) / abs((imfs[n][-1] - imfs[n][-2])) # also, extend is the best way
            else: # if there are more than two extremas
                amplitude_upper_ema = ema(extrema_upper_value, alpha=0.6)# whether use ema is a good thing here?
                amplitude_lower_ema = ema(extrema_lower_value, alpha=0.6)# whether use ema is a good thing here?
                nextremas = min(len(extrema_lower_index), len(extrema_upper_index))
                distance_set = abs(extrema_upper_index[-nextremas:] - extrema_lower_index[-nextremas:])
                distance = ema(distance_set, alpha=0.6)# here as well, not so sure whether ema is better though
                step = abs(amplitude_upper_ema - amplitude_lower_ema) / distance
                reference_amplitude = abs(amplitude_lower_ema) * 0.25 + abs(amplitude_upper_ema) * 0.25 + abs(
                        imfs[n][last_extrema]) * 0.5
                if imfs[n][-1]*imfs[n][last_extrema]<0: # if the last point has already crossed the axis
                    if abs(imfs[n][-1])>=0.8*reference_amplitude and abs(imfs[n][-1])+step>1.3*reference_amplitude:
                        forecast_value = imfs[n][-1]+step*(-abs(imfs[n][-1])/imfs[n][-1])
                    else:
                        forecast_value = reference_amplitude*(abs(imfs[n][-1])/imfs[n][-1])
                else:
                    forecast_value = imfs[n][-1] + step * (imfs[n][-1] - imfs[n][-2]) / abs((imfs[n][-1] - imfs[n][-2]))
                    if abs(forecast_value) >= abs(imfs[n][last_extrema])*1.1:
                        forecast_value = abs(imfs[n][last_extrema])*1.1*(-abs(imfs[n][-1])/imfs[n][-1])
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
        return forecast_value_vector, forecast_result

    return forecast_result

def ema(data, alpha): #simple function to give a ema as u want
        emaresult = data[0]
        for n in np.arange(len(data)-1):
            emaresult = emaresult*(1-alpha)+alpha*data[n+1]
        return emaresult

def dmddiffer(emasets, days_to_keep, days_to_use=15, pointsperday=288):
        dmds = []
        for n in np.arange(days_to_use):
            dmddataset = []
            dmdset = emasets[-days_to_use+n-days_to_keep:-days_to_use+n]
            for loop in np.arange(len(dmdset)):
                dmddataset.extend(dmdset[loop])
            dmddataset = np.array(dmddataset)
            hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=288).fit(dmddataset)
            hodmd.reconstructed_data.shape
            hodmd.dmd_time['tend'] = (days_to_keep + 1) * pointsperday - 1
            dmddataset = hodmd.reconstructed_data[0].real[-pointsperday:]
            dmds.append(dmddataset)
        return dmds

def egp(dataset):
    emd = EMD()
    imfs = emd(dataset)  # do the EMD
    nimfs = len(imfs)
    m=[]
    result =[]
    for n in np.arange(nimfs):
        imf = imfs[n]
        m.append(gp(imf))
        result.append(m[n].predict_noiseless(np.array([[len(imf)]]))[0][0][0])
    return result, sum(result)

def gp(dataset, draw=1):
    x = np.arange(len(dataset))
    GPy.plotting.change_plotting_library('matplotlib')
    kernel_1 = GPy.kern.PeriodicExponential(input_dim=1, variance=1., lengthscale=1.)
    kernel_2 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    kernel_3 = GPy.kern.sde_StdPeriodic(input_dim=1, variance=1., lengthscale=1.)
    kernel = kernel_1 + kernel_2 + kernel_3
    x.shape = (len(x), 1)
    dataset.shape = (len(dataset), 1)
    m = GPy.models.GPRegression(x, dataset, kernel)
    m.optimize(messages=False)
    if draw == 1:
        fig = m.plot(plot_density=False, figsize=(14, 6), dpi=300)
        GPy.plotting.show(fig, filename='basic_gp_regression_good luck')
        plt.show()
    return m



def gp_prediction(differsets): # the gp prediction
    emd = EMD()
    forecast_result = []
    for loop in np.arange(len(differsets)):
        imfs = emd(differsets[loop])  # do the EMD
        nimfs = len(imfs)
        forecast_value_vector = []
        for n in np.arange(nimfs):
            imf = imfs[n]
            imf.shape = (len(imf), 1)
            x = np.arange(len(imf))
            kernel_1 = GPy.kern.PeriodicExponential(input_dim=1, variance=1., lengthscale=1.)
            kernel_2 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
            kernel_3 = GPy.kern.sde_StdPeriodic(input_dim=1, variance=1., lengthscale=1.)

            kernel = kernel_1 + kernel_2+kernel_3
            x.shape = (len(x), 1)
            m = GPy.models.GPRegression(x, imf, kernel)
            m.optimize(messages=False)
            forecast_value = m.predict_noiseless(np.array([[len(imf)]]))[0][0][0] #get the prediction value
            forecast_value_vector.append(forecast_value)
        forecast_result.append((sum(forecast_value_vector)))
    return forecast_result

def gaojier(differsets):
    forecast_result = []
    for loop in np.arange(len(differsets)):
        data = differsets[loop]
        fittedmodel = auto_arima(data, start_p=1, start_q=1, max_p=6, max_q=6, max_d=6, max_order=None,
                                     seasonal=False, m=1, test='adf', trace=False,
                                     error_action='ignore',  # don't want to know if an order does not work
                                     suppress_warnings=True,  # don't want convergence warnings
                                     stepwise=True, information_criterion='bic', njob=-1)  # set to stepwise
        y_hat = fittedmodel.predict(1)[0]
        forecast_result.append(y_hat)
    return forecast_result

def datawash(differset): #wash the data, remove outliers
    for n in np.arange(len(differset)):
        Percentile = np.percentile(differset[n], [0, 25, 50, 75, 100])
        IQR = Percentile[3] - Percentile[1]
        UpLimit = Percentile[3] + IQR * 1.5
        DownLimit = Percentile[1] - IQR * 1.5
        differset[n][np.where(differset[n] > UpLimit)] = UpLimit
        differset[n][np.where(differset[n] < DownLimit)] = DownLimit
    return differset

def directions(data):# make only the directions work for forecast results
    data[np.where(data>0)] = 0.1
    data[np.where(data<0)] = -0.1
    return data






