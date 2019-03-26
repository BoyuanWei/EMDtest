# a pak to give the evaluation to the regression
# also a pak to give "prediction" on the random parts
from __future__ import division
import numpy as np
from math import sqrt
from PyEMD import EMD
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema as extrema
from pydmd import HODMD
import GPy
from pyramid.arima import auto_arima
from IPython.display import display
from scipy import stats

from sklearn.linear_model import BayesianRidge, LinearRegression

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
    imfset = []
    imfsumset = []
    for loop in np.arange(len(differsets)):
        imfs = emd(differsets[loop]) # do the EMD
        nimfs = len(imfs)
        imfset.append(imfs[0])
        extrema_upper_index_vector = []# record, make no sense
        extrema_lower_index_vector = []
        forecast_value_vector = []
        imfsums = []
        for n in np.arange(nimfs):
            # try to figure out the trend and give prediction
            #---------wash the extremas----------------------------------------------------
            extrema_upper_index = extrema(imfs[n], np.greater_equal)[0] # max extrema
            neighbours = []
            imfsums.append(np.sum(imfs[n]))
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
        imfsumset.append(imfsums)
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

    return forecast_result, imfset, imfsumset

def ema(data, alpha): #simple function to give a ema as u want
        emaresult = data[0]
        for n in np.arange(len(data)-1):
            emaresult = emaresult*(1-alpha)+alpha*data[n+1]
        return emaresult

def dmddiffer(emasets, days_to_keep, days_to_use=25, pointsperday=288): #get the multiple dmd result
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
    data[np.where(data > 0)] = 0.1
    data[np.where(data < 0)] = -0.1
    return data

def percents(forecasterror, realerror, dmdset): # to see the correction rate of the forecast
    nonzeros = np.where(dmdset > 0.01)[0]
    forecasterror_nonzeros = forecasterror[nonzeros]
    realerror_nonzeros = realerror[nonzeros]
    forecasterror_nonzeros[np.where(forecasterror_nonzeros > 0)] = 1
    forecasterror_nonzeros[np.where(forecasterror_nonzeros < 0)] = -1
    realerror_nonzeros[np.where(realerror_nonzeros > 0)] = -1
    realerror_nonzeros[np.where(realerror_nonzeros < 0)] = 1
    trial = forecasterror_nonzeros+realerror_nonzeros
    success = np.array(np.where(abs(trial) == 0))
    fail = np.array(np.where(abs(trial) == 2))
    rate = len(success[0])/(len(fail[0])+len(success[0]))
    action_rate = (len(fail[0])+len(success[0]))/len(trial)
    return rate, action_rate, trial, nonzeros

def hour(data):
    days=int(np.floor(len(data)/2))
    data_new = []
    for n in np.arange(days):
        data_new.append(sum(data[n*2:(n+1)*2]))
    return np.array(data_new)
def changes(differset):#to show the distribution of next step from the current step.
    zerodotfive = []
    one = []
    onedotfive = []
    two = []
    twodotfive = []
    three = []
    threedotfive = []
    four=[]
    fourdotfive = []
    fiveplus = []
    mzerodotfive = []
    mone = []
    monedotfive = []
    mtwo = []
    mtwodotfive = []
    mthree = []
    mthreedotfive = []
    mfour = []
    mfourdotfive = []
    mfiveplus = []
    nums = len(differset)
    for n in np.arange(nums):
        length = len(differset[n])
        for loop in np.arange(length-1):
            if differset[n][loop] >= 5:
                fiveplus.append(differset[n][loop+1])
            elif differset[n][loop] >= 4.5:
                fourdotfive.append(differset[n][loop+1])
            elif differset[n][loop] >= 4:
                threedotfive.append(differset[n][loop+1])
            elif differset[n][loop] >= 3.5:
                three.append(differset[n][loop+1])
            elif differset[n][loop] >= 3:
                three.append(differset[n][loop+1])
            elif differset[n][loop] >= 2.5:
                twodotfive.append(differset[n][loop+1])
            elif differset[n][loop] >= 2:
                two.append(differset[n][loop+1])
            elif differset[n][loop] >= 1.5:
                onedotfive.append(differset[n][loop+1])
            elif differset[n][loop] >= 1:
                one.append(differset[n][loop+1])
            else:
                pass
    return fiveplus, fourdotfive, four, threedotfive, three, twodotfive, two, onedotfive, one

def bayfit(data): #Bayesian regression - very bad
    lw = 2
    x = np.arange(len(data)-1)
    degree = 3
    clf_poly = BayesianRidge()
    clf_poly.fit(np.vander(x, degree), data[x])
    x_plot = np.arange(len(data))
    y_mean, y_std = clf_poly.predict(np.vander(x_plot, degree), return_std=True)
    plt.figure(figsize=(6, 5))
    plt.errorbar(x_plot, y_mean, y_std, color='navy',
                 label="Polynomial Bayesian Ridge Regression", linewidth=lw)
    plt.plot(x_plot, data, color='gold', linewidth=lw,
             label="Ground Truth")
    plt.ylabel("Output y")
    plt.xlabel("Feature X")
    plt.legend(loc="lower left")
    plt.show()

def baynetwork(differset, dmdset): # to get the statistic stuff on the last five errors and the last one
    nonzeros = np.where(dmdset > 0.01)[0]
    posterior = [0]*12
    count = [0]*12
    for loop in nonzeros:
        stand = 0# np.mean(differset[loop])
        for n in np.arange(len(differset[loop])-5):
            upper = np.where(differset[loop][n:n+5] > stand)[0]
            lower = np.where(differset[loop][n:n+5] < stand)[0]
            k = len(upper)-len(lower)
            if differset[loop][n+4] > stand:
                place = 1
            else:
                place = 0
            count[k + 5+place] = count[k + 5+place] + 1
            if differset[loop][n+5] > stand:
                posterior[k+5+place] = posterior[k+5+place]+1
            elif differset[loop][n+5] < stand:
                posterior[k+5+place] = posterior[k+5+place]-1
            else:
                pass
    return posterior, count

def bino(baynetworkresult, differset, dmdset):
    nonzeros = np.where(dmdset > 0.01)[0]
    forecast = [0]*len(differset)
    rate = []
    #stage = 0

    for n in np.arange(len(baynetworkresult[0])):
        if baynetworkresult[1][n] == 0:
            rate.append(0)
        else:
            p = baynetworkresult[0][n]/baynetworkresult[1][n]
            rate.append((1.4+p)*p)
    rate = np.array(rate)
    rate[np.where(rate > 1)] = 1
    rate[np.where(rate < -1)] = -1
    for loop in nonzeros:
        standard = 0#np.mean(differset[loop])
        upper = np.where(differset[loop][-5:] > standard)[0]
        lower = np.where(differset[loop][-5:] < standard)[0]
        #ma = ema(abs(differset[loop]), 0.5)
        k = len(upper) - len(lower)
        if differset[loop][-1] > standard:
            place = 1
        else:
            place = 0
        prob = rate[k+5+place]
        if prob > 0:
            ma = ema(abs(differset[loop][np.where(differset[loop] > standard)]), 0.5)
            forecast[loop] = prob*ma+standard
        elif prob < 0:
            ma = ema(abs(differset[loop][np.where(differset[loop] < standard)]), 0.5)
            forecast[loop] = prob*ma+standard
    return forecast

def bayesianstatistic(differset, dmdset): # do the statisitic stuff to see whether count the first 3 and combine the last 2 is better.
    nonzeros = np.where(dmdset > 0.01)[0]
    posterior = [0] * 16
    count = [0] * 16
    stand = 0# the "0"
    for loop in nonzeros:
        for n in np.arange(len(differset[loop])-5):
            upper = np.where(differset[loop][n:n+3] > stand)[0]
            #lower = np.where(differset[loop][n:n+3] < stand)[0]
            k = len(upper)*4
            if differset[loop][n+3] <0:
                if differset[loop][n+4] < 0 :
                    place = 0
                else:
                    place = 1
            else:
                if differset[loop][n+4] < 0 :
                    place = 2
                else:
                    place = 3
            count[k + place] = count[k + place] + 1
            if differset[loop][n+5] > stand:
                posterior[k+place] = posterior[k+place]+1
            elif differset[loop][n+5] < stand:
                posterior[k+place] = posterior[k+place]-1
            else:
                pass
    return posterior, count, np.array(posterior)/np.array(count)

def bsbino(baynetworkresult, differset, dmdset):
    nonzeros = np.where(dmdset > 0.01)[0]
    forecast = [0]*len(differset)
    rate = []
    #stage = 0

    for n in np.arange(len(baynetworkresult[0])):
        if baynetworkresult[1][n] == 0:
            rate.append(0)
        else:
            p = baynetworkresult[0][n]/baynetworkresult[1][n]
            rate.append(p)
    rate = np.array(rate)
    rate[np.where(rate > 1)] = 1
    rate[np.where(rate < -1)] = -1
    for loop in nonzeros:
        standard = 0#np.mean(differset[loop])
        upper = np.where(differset[loop][-5:-2] > standard)[0]
        # lower = np.where(differset[loop][n:n+3] < stand)[0]
        k = len(upper) * 4
        if differset[loop][-2] < 0:
            if differset[loop][-1] < 0:
                place = 0
            else:
                place = 1
        else:
            if differset[loop][-1] < 0:
                place = 2
            else:
                place = 3
        prob = rate[k+place]
        if prob > 0:
            ma = ema(abs(differset[loop][np.where(differset[loop] > standard)]), 0.5)
            forecast[loop] = prob*ma+standard
        elif prob < 0:
            ma = ema(abs(differset[loop][np.where(differset[loop] < standard)]), 0.5)
            forecast[loop] = prob*ma+standard
    return forecast

def emddmd(dataset,d,draw=0): #can we do the dmd again? to see whether there are some miracles
    emd = EMD()
    imfs = emd(dataset)  # do the EMD
    nimfs = len(imfs)
    result = []
    sumresult = []
    #firstimf = firstone([imfs[0]])
    #result.append(firstimf)
    for n in np.arange(nimfs):
        dmddataset = imfs[n]
        hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=d).fit(dmddataset)
        hodmd.reconstructed_data.shape
        hodmd.dmd_time['tend'] = len(imfs[n])+24
        dmdresult = hodmd.reconstructed_data[0].real
        #uppercap = np.max(dmddataset)*1.1
        #lowercap = np.min(dmddataset)*1.1
        #if dmdresult[-2] > uppercap:
           # dmdresult[-2] = uppercap
        #elif dmdresult[-2] < lowercap:
            #dmdresult[-2] = lowercap

        if draw ==1:
            fig = plt.figure(figsize=(20, 10))
            plt.plot(np.arange(len(imfs[n])), imfs[n], '-', label='the practical signal',
                        color='g')
            plt.plot(np.arange(len(imfs[n])+25), dmdresult, '--', label='DMD output', color='r')
            plt.show()
        result.append(dmdresult[-2])
    sumresult.append(np.sum(result))
    return sumresult, result

def emddmdp(differset, dmdset):# just try whether dmd can really save me
    nonzeros = np.where(dmdset > 0.01)[0]
    forecast = [0] * len(differset)
    for n in nonzeros:
        forecast[n] = np.array(emddmd(differset[n], 12)[0])
    return forecast

def firstone(imfs,draw=0):
    extrema_upper_index_vector = []  # record, make no sense
    extrema_lower_index_vector = []
    forecast_value_vector = []
    imfsums = []
    for n in np.arange(1):  ##########################################################################
        # try to figure out the trend and give prediction
        # ---------wash the extremas----------------------------------------------------
        extrema_upper_index = extrema(imfs[n], np.greater_equal)[0]  # max extrema
        neighbours = []
        imfsums.append(np.sum(imfs[n]))
        for i in np.arange(len(extrema_upper_index) - 1):  # clean the indexes which close to each other
            if extrema_upper_index[i] - extrema_upper_index[i + 1] == -1:
                neighbours.append(i)
        extrema_upper_index = np.delete(extrema_upper_index, neighbours)
        extrema_upper_index = np.delete(extrema_upper_index, np.where((extrema_upper_index == 0) |
                                                                      (extrema_upper_index == len(imfs[n]) - 1)))
        neighbours = []

        extrema_lower_index = extrema(imfs[n], np.less_equal)[0]  # min exrema
        for i in np.arange(len(extrema_lower_index) - 1):  # clean the indexes which close to each other
            if extrema_lower_index[i] - extrema_lower_index[i + 1] == -1:
                neighbours.append(i)
        extrema_lower_index = np.delete(extrema_lower_index, neighbours)
        extrema_lower_index = np.delete(extrema_lower_index, np.where((extrema_lower_index == 0) |
                                                                      (extrema_lower_index == len(imfs[n] - 1) - 1)))
        if draw == 1:
            extrema_upper_index_vector.append(extrema_upper_index)
            extrema_lower_index_vector.append(extrema_lower_index)

        # ------------------------ the derivation starts from here---------------------

        # --some basic calculations --------#
        extrema_upper_value = imfs[n][extrema_upper_index]
        extrema_lower_value = imfs[n][extrema_lower_index]
        extremas = np.unique(np.hstack([extrema_upper_index, extrema_lower_index]))
        if extremas.any():
            last_extrema = extremas[-1]
        else:
            last_extrema = len(imfs[n]) - 1
        if len(extrema_upper_index) + len(extrema_lower_index) <= 0:  # if there is no real extrema
            distance = last_extrema  # means that there is no enough extremas to do the calculation
            amplitude_upper_ema = max(imfs[n])
            amplitude_lower_ema = min(imfs[n])
            step = abs(amplitude_upper_ema - amplitude_lower_ema) / distance
            forecast_value = imfs[n][-1] + step * (imfs[n][-1] - imfs[n][-2]) / abs(
                (imfs[n][-1] - imfs[n][-2]))  # just extend the tread
        elif len(extrema_upper_index) + len(extrema_lower_index) == 1:  # if there is only one extrema
            distance = len(imfs[n]) - last_extrema
            amplitude_upper_ema = max(imfs[n][last_extrema], imfs[n][-1])
            amplitude_lower_ema = min(imfs[n][last_extrema], imfs[n][-1])
            step = abs(amplitude_upper_ema - amplitude_lower_ema) / distance
            # reference_amplitude = abs(imfs[n][-1]) + 2 * step
            forecast_value = imfs[n][-1] + step * (imfs[n][-1] - imfs[n][-2]) / abs(
                (imfs[n][-1] - imfs[n][-2]))  # also, extend is the best way
        else:  # if there are more than two extremas
            amplitude_upper_ema = ema(extrema_upper_value, alpha=0.6)  # whether use ema is a good thing here?
            amplitude_lower_ema = ema(extrema_lower_value, alpha=0.6)  # whether use ema is a good thing here?
            nextremas = min(len(extrema_lower_index), len(extrema_upper_index))
            distance_set = abs(extrema_upper_index[-nextremas:] - extrema_lower_index[-nextremas:])
            distance = ema(distance_set, alpha=0.6)  # here as well, not so sure whether ema is better though
            step = abs(amplitude_upper_ema - amplitude_lower_ema) / distance
            reference_amplitude = abs(amplitude_lower_ema) * 0.25 + abs(amplitude_upper_ema) * 0.25 + abs(
                imfs[n][last_extrema]) * 0.5
            if imfs[n][-1] * imfs[n][last_extrema] < 0:  # if the last point has already crossed the axis
                if abs(imfs[n][-1]) >= 0.8 * reference_amplitude and abs(
                        imfs[n][-1]) + step > 1.3 * reference_amplitude:
                    forecast_value = imfs[n][-1] + step * (-abs(imfs[n][-1]) / imfs[n][-1])
                else:
                    forecast_value = reference_amplitude * (abs(imfs[n][-1]) / imfs[n][-1])
            else:
                forecast_value = imfs[n][-1] + step * (imfs[n][-1] - imfs[n][-2]) / abs((imfs[n][-1] - imfs[n][-2]))
                if abs(forecast_value) >= abs(imfs[n][last_extrema]) * 1.1:
                    forecast_value = abs(imfs[n][last_extrema]) * 1.1 * (-abs(imfs[n][-1]) / imfs[n][-1])
    return forecast_value
#def nbayesian():# see whether there is a possibility to apply the naive bayesian.
def getfirstone(differset, differ, n,draw=0):#get forecasts for the rest of the imfs and then derive how much the forecast on the first order should be
    rest_signal = emddmd(differset, 12)
    fistone = differ[n]-rest_signal[0]
    if draw==1:
        emd=EMD()
        imf_1=emd(differset)[0]
        fig = plt.figure(figsize=(20, 10))
        plt.plot(np.arange(len(imf_1)), imf_1, '-', label='the imf',
                 color='g')
        plt.scatter(len(imf_1)+1, fistone, marker='o', c='black', s=50)
        plt.show()
    return fistone

def baydmd(differset, dmddata):# do dmd on the rest and do bayesian on the first imf.
    otherresult = emddmdp(differset, dmddata) # do dmd on the rest
    emd = EMD()
    nonzeros = np.where(dmddata > 0.01)[0]
    forecast = [0] * len(differset)
    imfs = [0]*len(differset)
    for n in nonzeros:
        imf_1 = emd(differset[n])[0]
        imfs[n] = imf_1
    #do the statistical stuff:
    stat_result = baynetwork(imfs, dmddata)
    first_imf_forecast = bino(stat_result, imfs, dmddata)
    final_forecast = np.array(otherresult)+np.array(first_imf_forecast)
    return final_forecast





















