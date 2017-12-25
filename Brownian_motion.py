import math
import numpy as np
from numpy import random as npr
import scipy
import matplotlib
from matplotlib import pyplot as plt


def brownian_motion(dt=0,x0=0, N=1000):
    N = scipy.zeros(N+1)
    'wE CREATE N+1 timestes'

    t = scipy.linspace(0, N, N+1)

    W[1:N+1] = scipy.cumsum(scipy.random.normal(0, dt, N))
    return t, N


def plot_brownian(t, w):
    plt.plot(t, w)
    plt.xlabel("Time(t)")
    plt.ylabel("Winer_process w(t)")
    plt.title("Winers-process")
    plt.show()


t, w = brownian_motion()
plot_brownian(t, w)


