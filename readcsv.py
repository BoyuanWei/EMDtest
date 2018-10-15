#This is used to read the PV generation data report from enegerville system
import pandas as pd
import numpy as np
def rd(dataname, datawashflag=1):
    df = pd.read_csv(dataname, header=8, names=['time', 'data'],
                     dtype={'time': str, 'data': np.float})
    if datawashflag == 1:
        df[df.data > 30] = 30
        df[df.data < 0] = 0
    dataarray = df.data[:-1]
    dataarray = np.array(dataarray)
    nanindex = np.where(np.isnan(dataarray))
    for n in nanindex[0]:
        frontier_flag = 0
        nanpos_upper = n
        while np.isnan(dataarray[nanpos_upper]):
            if nanpos_upper < len(dataarray):
                nanpos_upper = nanpos_upper + 1
            else:
                frontier_flag = 1

        nanpos_lower = n
        while np.isnan(dataarray[nanpos_lower]):
            if nanpos_lower > 1:
                nanpos_lower = nanpos_lower-1
            else:
                frontier_flag = 1
        if frontier_flag == 1:
            dataarray[n] = 0
        else:
            dataarray[n] = np.mean([dataarray[nanpos_upper], dataarray[nanpos_lower]])
    return dataarray