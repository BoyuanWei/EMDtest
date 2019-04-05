# the prediction by dmd with removing the zero zones
import matplotlib.pyplot as plt
from readdata import read as rd
from termcolor import *
from pydmd import HODMD
import numpy as np
import readcsv
from math import sqrt
from evaluation import ev as ev
from plotcheck import pl

import termcolor

from evaluation import fields
from evaluation import zeromask



#windset = rd(path)
#name = raw_input('the name of data set?')
#realwindset = windset[name]
pointsperday = 288 # CHANGE HERE FOR DIFFERENT RESOLUTION
#--------------------swithes---------------
moving_average_flag = 0  # ------------------CHANGE WHETHER THE MOVING AVERAGE IS NEEDED HERE.--------------
zero_zone_cut = 1# ------------------CHANGE WHETHER THE ZERO ZONE CUT IS NEEDED HERE.--------------
stages_reference = 1# ------------------CHANGE WHETHER USE THE DATA FROM SEVERAL YEARS AGO AS REFERENCE.--------------
zero_mask_flag = 1#-----------------------CHANGE WHETHER ZEROS ARE MASKED-----------------------------
combination = 1 #--------CHANGE WHETHER USE THE COMBINATION AMONG WITH MV, WITHOUT MV, SEVERAL YEARS AGO--------



#--------------- read the old data sets-------------
if stages_reference == 1:
    path_2016 = "/home/bwei/PycharmProjects/data lib/PV2016_apr.csv"
    dataset_2016 = readcsv.rd(path_2016)
    #dataset_2016[np.where(dataset_2016>50)]=0
#read_2017:
    path_2017 = "/home/bwei/PycharmProjects/data lib/PV2017_apr.csv"
    dataset_2017 = readcsv.rd(path_2017)
    #dataset_2017[np.where(dataset_2017>50)]=0
#read_2018:

    path_2018 = "/home/bwei/PycharmProjects/data lib/PV2018_apr.csv"
    dataset_2018 = readcsv.rd(path_2018)
realwindset = dataset_2018
#realwindset=hour(realwindset)#############
windsetoriginal= realwindset
realwindset.shape = (len(realwindset),)

#data reading is done

def ma(data, days_to_keep, points_per_day, alpha=0.25 ):
    days_covered = int(np.floor(len(data)/points_per_day))
    points_covered = days_covered*points_per_day
    daysdata = []
    onedaydata = data[len(data)-points_covered:len(data)-points_covered+points_per_day]
    for loop in np.arange(days_covered-1):
        onedaydata = onedaydata*(1-alpha)+alpha*data[len(data)-points_covered+points_per_day*(loop+1):
                                                         len(data)-points_covered+points_per_day*(loop+1)+points_per_day]
        daysdata.append(onedaydata)
    return daysdata[-days_to_keep:], daysdata

def non2full(dmd_prediction, zero_data_index, pointsperday):
    full_prediction_one_day = np.array([0.00]*pointsperday) # restore the nonzero prediction result to a full day
    position_flag = 0
    for n in np.arange(pointsperday):
        if n in zero_data_index:
            pass
        else:
            full_prediction_one_day[n] = dmd_prediction[position_flag]
            position_flag = position_flag+1
    return full_prediction_one_day

#-----core prediction---------------#
def predicition(dmddataset, days):
    hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=len(dmddataset)/days).fit(dmddataset)
    hodmd.reconstructed_data.shape
    hodmd.plot_eigs()
    hodmd.dmd_time['tend'] = len(dmddataset)/days*(days+1)-1# since it starts from zero
    dmd_output = hodmd.reconstructed_data[0].real
    dmd_prediction = dmd_output[-len(dmddataset)/days:]
    return dmd_prediction, dmd_output


nmae_record = []

for loop in np.arange(66, 150):

    days = 4
    cut = loop

    cut = cut*pointsperday
    dmddataset = realwindset[cut-days*pointsperday:cut]

#------- moving average process------------------
    if combination == 1:
        moving_average_flag = 1 # force moving average flag on
        dmddataset_org = dmddataset # save the original dataset for later use

    if moving_average_flag == 1:
        day_ahead = 10
        data_for_ma = realwindset[cut - pointsperday * day_ahead:cut]
        moving_average_set = ma(data_for_ma, days, pointsperday)[0]
        dmddataset = []
        for n in np.arange(len(moving_average_set)):
            dmddataset.extend(moving_average_set[n])
        dmddataset = np.array(dmddataset)

    # --------clean the zero zone---------
    if zero_zone_cut == 1:
        day_ahead = 10
        data_for_ma = realwindset[cut - pointsperday * day_ahead:cut]
        dmddataset_ma = ma(data_for_ma, days, pointsperday)[0][-1]
        zero_data_index = np.where(dmddataset_ma == 0)
        zero_data_index_org = zero_data_index
        for n in np.arange(days-1):
            zero_data_index = np.append(zero_data_index, zero_data_index+(n+1)*pointsperday)
        dmddataset = np.delete(dmddataset, zero_data_index)
        if 'dmddataset_org' in locals().keys():
            dmddataset_org = np.delete(dmddataset_org, zero_data_index)

    if combination == 1: # do the main work: prediction
        dmd_prediction_main = predicition(dmddataset, days)
        dmd_output = dmd_prediction_main[1] # just give the dmdoutput some thing to be draw
        dmd_short = predicition(dmddataset_org[-len(dmddataset_org)/days*3:], 3)
        dmd_prediction_short = dmd_short[0]
        # Check whether the short prediction is trustworthy
        dmd_short_lastday = dmd_short[1][-len(dmd_prediction_short)*2:-len(dmd_prediction_short)]
        dmddataset_org_lastday = dmddataset_org[-len(dmddataset_org)/days:]
        dmd_short_lastday_stage = fields(dmd_short_lastday)
        dmddataset_org_lastday_stage = fields(dmddataset_org_lastday)
        if abs(dmd_short_lastday_stage - dmddataset_org_lastday_stage) < 1:
            short_trustworthy_flag = 1
        else:
            short_trustworthy_flag = 0
            print(colored('Short dynamic capture failed!', 'green'))
        if zero_mask_flag == 1:
            dmd_prediction_short[np.where(dmd_prediction_short < 0)] = 0
        dmd_prediction = dmd_prediction_main[0]
    else:
        dmd_result = predicition(dmddataset, days)
        dmd_prediction = dmd_result[0]
        dmd_output = dmd_result[1]
        if zero_mask_flag == 1:
            dmd_prediction[np.where(dmd_prediction < 0)] = 0


    #----the place to implement stages correction
    if stages_reference == 1:
        stage_2016 = fields(dataset_2016[cut:cut+pointsperday])
        stage_2017 = fields(dataset_2017[cut:cut+pointsperday])
        dmd_prediction[np.where(dmd_prediction < 0)] = 0  # remove the negative forcast
        stage_prediction = fields(dmd_prediction)
        stage_practical = fields(realwindset[cut:cut+pointsperday])
        if combination == 1:
            stage_short_prediction = fields(dmd_prediction_short)

    if combination == 1:

        #opnions:
        opinion_yearly = stage_2017*0.65+stage_2016*0.35 # the opinion from yearly correlation
        opinion_short = stage_short_prediction # the opinion from shortly forecast
        #corrections
        if short_trustworthy_flag == 1 and abs(opinion_yearly-opinion_short) >= 1:
            stage_correction = np.floor((opinion_short-opinion_yearly)/abs(opinion_short-opinion_yearly)) # how the stage should be corrected
            suggested_stage = stage_correction+np.floor(opinion_yearly)
            correction_direction = np.floor((suggested_stage-stage_prediction)/abs(suggested_stage-stage_prediction))
            correction_flag = 1 # set the correlation flag
            dmd_prediction_short_nominal = abs(dmd_prediction_short / np.max(abs(dmd_prediction_short))) # nominalize
            new_stage = fields(dmd_prediction)
            while np.floor(new_stage) != suggested_stage:
                dmd_prediction = dmd_prediction + dmd_prediction_short_nominal*correction_direction
                dmd_prediction[np.where(dmd_prediction < 0)] = 0
                new_stage = fields(dmd_prediction)
        elif short_trustworthy_flag == 1:
            if (stage_prediction-opinion_yearly)*(stage_prediction-opinion_short) < 0:
                pass
            else:
                correction_flag = 2
                correction_direction = np.floor((opinion_short - stage_prediction) / abs(opinion_short - stage_prediction))
                dmd_prediction_short_nominal = abs(dmd_prediction_short / np.max(abs(dmd_prediction_short)))  # nominalize
                new_stage = stage_prediction
                while (new_stage-opinion_yearly)*(new_stage-opinion_short)>0:
                    dmd_prediction = dmd_prediction + dmd_prediction_short_nominal * correction_direction
                    dmd_prediction[np.where(dmd_prediction < 0)] = 0 #negative proof
                    new_stage = fields(dmd_prediction)
                    if abs(opinion_yearly-opinion_short) < 0.2:
                        if abs(new_stage-opinion_short) < 0.2 or abs(new_stage-opinion_yearly) < 0.2:
                            break
        else:
            if abs(stage_prediction-opinion_yearly) >= 1:
                correction_flag = 3
                stage_correction = np.floor((opinion_yearly - stage_prediction) / abs(opinion_yearly - stage_prediction))
                new_stage = stage_prediction
                while abs(new_stage-opinion_yearly) >= 1:
                    dmd_prediction = dmd_prediction*(1+0.05*stage_correction)
                    dmd_prediction[np.where(dmd_prediction < 0)] = 0  # negative proof
                    new_stage = fields(dmd_prediction)



    if zero_zone_cut == 1 :
        full_prediction_one_day = non2full(dmd_prediction, zero_data_index_org[0], pointsperday)
    else:
        full_prediction_one_day = dmd_prediction

#----one pic about the with DMD output





#Start the error evaluation section from here(:

    data_prediction_dmd = full_prediction_one_day
    data_practical_eva = realwindset[cut:(cut+pointsperday)]
    ev_result = ev(data_prediction_dmd, data_practical_eva)
    nmae = ev_result['MAE']/np.max(dataset_2018)*100
    nmae_record.append(nmae)
    print loop

nmae_record = np.array(nmae_record)

pl(nmae_record)









