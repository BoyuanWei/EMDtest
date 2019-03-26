import matplotlib.pyplot as plt
import numpy as np

def pl(data):
    size = len(data)
    x = np.arange(0, size, 1)
    fig = plt.figure(figsize=(15, 5))
    plt.plot(x, data, '-', label='data')
    mid = np.mean(data)
    plt.hlines(mid, 0, size, colors="c", linestyles="dashed", label='mean')
    plt.legend()
    plt.show()
    return()