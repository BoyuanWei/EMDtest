import os
from scipy import io
def read(path):
    data={}
    files = os.listdir(path)
    for name in files:
      path2 = path+"/"+name
      dataraw = io.loadmat(path2)
      datarawab = {name[:-4]:dataraw[name[:-4]]}
      data = dict(data, **datarawab)
    return data
