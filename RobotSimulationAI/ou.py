import numpy
from scipy import stats
import torch
def mean(x0, t, mean_rev_speed, mean_rev_level):
    assert mean_rev_speed >= 0
    return x0 * numpy.exp(-mean_rev_speed * t) + (1.0 - numpy.exp(- mean_rev_speed * t)) * mean_rev_level

def variance(t, mean_rev_speed, vola):
    assert mean_rev_speed >= 0
    assert vola >= 0
    return vola * vola * (1.0 - numpy.exp(- 2.0 * mean_rev_speed * t)) / (2 * mean_rev_speed)

def std(t, mean_rev_speed, vola):
    return numpy.sqrt(variance(t, mean_rev_speed, vola))
def path(x0, t, mean_rev_speed, mean_rev_level, vola):
    """ Simulates a sample path"""
    assert len(t) > 1
    x = stats.norm.rvs(size=len(t))
    x[0] = x0
    dt = numpy.diff(t)
    scale = std(dt, mean_rev_speed, vola)
    x[1:] = x[1:] * scale
    for i in range(1, len(x)):
        x[i] += mean(x[i - 1], dt[i - 1], mean_rev_speed, mean_rev_level)
    return x
class OUNoise:
    def __init__(self,x0, dt, mean_rev_speed, mean_rev_level, vola):
        self.x = x0
        self.dt = dt
        self.scale = std(dt, mean_rev_speed, vola)
        self.mean_rev_speed = mean_rev_speed
        self.mean_rev_level = mean_rev_level
    def __next__(self):
        x = stats.norm.rvs(size=1)*self.scale
        x += mean(self.x, self.dt, self.mean_rev_speed, self.mean_rev_level)
        self.x = x
        return x[0]
class ND_OUNoise:
    def __init__(self,n,x0, dt, mean_rev_speed, mean_rev_level, vola):
        self.noises = [OUNoise(x0, dt, mean_rev_speed, mean_rev_level, vola) for _ in range(n)]
    def __next__(self):
        return torch.tensor([noise.__next__() for noise in self.noises])