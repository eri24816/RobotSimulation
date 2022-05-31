from collections import defaultdict
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
class Stat:
    def __init__(self,auto_log_n_iter = -1,log_dir = None):
        self.sum = defaultdict(float)
        self.denom = defaultdict(float)
        self.xs = []
        self.rec = defaultdict(list)
        self.log_iter_target = auto_log_n_iter
        self.log_iter = 0
        self.epoch = 0

        if log_dir is None:
            log_dir = datetime.now().strftime("%m_%d_%Y/%H_%M_%S")
        self.writer = SummaryWriter(log_dir=os.path.join('runs/',log_dir))
        
    def add(self,name,value,auto_flush = True):
        if torch.is_tensor(value):
            value = value.detach()
        self.sum[name]+=value
        self.denom[name]+=1
        
        self.log_iter +=1
        if auto_flush and self.log_iter>=self.log_iter_target:
            self.log()
        
    def flush(self):
        for k in self.sum.keys():
            if self.denom[k]==0:
                continue
            mean_value = self.sum[k]/self.denom[k]
            self.rec[k].append(mean_value)
            self.writer.add_scalar(k,mean_value,self.epoch)
        self.xs.append(self.epoch)
        
        self.sum = defaultdict(float)
        self.denom = defaultdict(float)

        
    def log(self):
        res = ''
        for k in self.sum.keys():
            if self.denom[k]==0:
                continue
            res+=f"{k}: {self.sum[k]/self.denom[k]}\t"
        print(res)
        
        self.flush()
        self.log_iter = 0
        
    def plot(self,name,start=0,end=None):
        plt.plot(self.xs[start:end],self.rec[name][start:end])
        
    def clear(self):
        self.rec = defaultdict(list)
        self.xs = []
        
    def set_epoch(self,value):
        self.epoch = value