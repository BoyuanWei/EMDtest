import os
from scipy import io


def read(path):
    data = {}
    files = os.listdir(path)
    for name in files:
        path2 = path+"/"+name
        data_raw = io.loadmat(path2)    # read the file
        data_raw_ary = {name[:-4]: data_raw[name[:-4]]}  # get the array from raw data
        data = dict(data, **data_raw_ary)   # put the new objective into dataset
    return data
