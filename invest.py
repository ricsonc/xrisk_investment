#!/usr/bin/env python

from ipdb import set_trace as st

from munch import Munch as M
from time import time
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Gumbel

def mc(x):
    #return x.cuda()
    return x

def gumbel_sigmoid(logp, t):
    log1mp = torch.log(1-torch.exp(logp))
    logit = logp-log1mp

    u1 = mc(torch.rand(logit.shape)).type(torch.float64)
    u2 = mc(torch.rand(logit.shape)).type(torch.float64)
    eps = 1E-12
    
    noise = -torch.log(torch.log(u2 + eps)/torch.log(u1 + eps) +eps)

    soft = torch.sigmoid((logit + noise) / t)
    hard = (soft > 0.5).type(torch.float64)

    #returns hard, but with the gradient of soft!
    #the order of operations here is crucial to prevent rounding errors...
    return hard.detach() + (soft - soft.detach())

class XRisk(nn.Module):
    def __init__(self, batch_size = 10000, temperature = 1000, record = False):

        super(XRisk, self).__init__()
        
        self.maxt = 10000
        
        self.log_gamma = np.log(1.02) 
        self.alpha = np.log(0.99)/1000.0

        self.initial_logw = np.log(0.248)
        self.initial_logp = np.log(0.001) 

        #as temperature -> 1, rewards approach original
        self.temperature = temperature
        self.gumbel_temp = 1
        
        self.batch_size = batch_size

        self.actions = nn.Parameter(-10.0 * mc(torch.ones(self.maxt, dtype = torch.float64)), requires_grad = True)

        self.record = record

    def get_reward(self):
        return self.alive * torch.exp(self.logw/self.temperature)
        
    def reset(self):

        self.history = M(logp = [], logw = [], logf = [], alive = [], rwd = [])
        
        self.logp = self.initial_logp * mc(torch.ones(self.batch_size, dtype = torch.float64))
        self.logw = self.initial_logw * mc(torch.ones(self.batch_size, dtype = torch.float64))
        self.t = mc(torch.zeros(self.batch_size, dtype = torch.int32))
        self.alive = mc(torch.ones(self.batch_size, dtype = torch.float64))

    def step(self, action):
        f = torch.exp(action)

        self.t += 1

        #replace with gumbel sampling
        self.alive = self.alive * (1.0 - gumbel_sigmoid(self.logp, self.gumbel_temp))

        logw = self.logw + self.log_gamma + torch.log(1.0 - f)
        logp = self.logp + self.alpha * f * torch.exp(self.logw)

        self.logw = logw
        self.logp = logp

        if self.record:
            detach_mean = lambda x: x.detach().cpu().numpy().mean()
            self.history.logp.append(detach_mean(self.logp))
            self.history.logw.append(detach_mean(self.logw))
            self.history.logf.append(action.detach().cpu().numpy())
            self.history.alive.append(detach_mean(self.alive))
            self.history.rwd.append(detach_mean(self.get_reward()))

    def forward(self, acts = None):

        if acts is None:
            acts = self.actions
            
        self.reset()
        cumrew = 0.0
        for i in range(self.maxt):
            if i%100 == 0:
                print(i)
            self.step(acts[i])
            cumrew = cumrew + self.get_reward()

        cumrew = cumrew.mean()

        return cumrew

    def plot(self, name):

        assert self.record
        
        fig, ax = plt.subplots(2, 3, figsize=(18, 12))

        fig.suptitle(name)
        
        ax[0,0].plot(np.exp(self.history.logp))
        ax[0,0].set_title('x-risk')
        ax[0,0].set_yscale('log')
        ax[0,0].set_ylim(10**-9, 1)

        ax[0,1].plot(np.exp(self.history.logw))
        ax[0,1].set_title('wealth')
        ax[0,1].set_yscale('log')
        
        ax[0,2].plot(np.exp(self.history.logf))
        ax[0,2].set_title('spend fraction')
        ax[0,2].set_yscale('log')

        ax[1,0].plot(self.history.alive)
        ax[1,0].set_title('alive')
        ax[1,0].set_yscale('log')

        ax[1,1].plot(self.history.rwd)
        ax[1,1].set_title('expected reward')
        ax[1,1].set_yscale('log')
        
        #plt.show()
        plt.savefig(name+'.png', dpi=200, bbox_inches='tight')

def get_probability_matcher_actions(start=0):
    task=XRisk(batch_size = 1, temperature=1.0)
    task.reset()
    actions = []
    for i in range(task.maxt):
        act = -1000.0 if i < start else task.logp.detach().numpy()[0]
        
        actions.append(act)
        task.step(torch.from_numpy(np.array([act])))

    return np.array(actions)
        
if __name__ == '__main__':

    if False:
        task = XRisk(temperature=1.0)
        optimizer = optim.Adam(task.parameters(), lr = 5e-2)

        for i in range(1000):
            t0 = time()

            score = task()
            (-score).backward()
            optimizer.step()
            optimizer.zero_grad()
            task.gumbel_temp = max(task.gumbel_temp * 0.95, 0.001)

            print ('iteration', i, score, 'time taken', time()-t0)

            if i % 20 == 0:
                actions = task.actions.detach().cpu().numpy()
                np.save('actions_%d' % i, actions)

        plt.plot(task.actions.detach().cpu().numpy())
        plt.show()
    
    if True:
        task = XRisk(temperature=1.0, batch_size = 1000000, record = True)


        #uncomment one of the below sections
        
        #constant investment
        # s = task(acts = torch.from_numpy(-1000.0 * np.ones(task.maxt, dtype = np.float64)))
        # task.plot('constant-investment')

        #learned 
        # i = 980
        # s = task(acts = torch.from_numpy(np.load('actions_%d.npy' % i)))
        # task.plot('learned')
        
        #constant reduction
        # s = task(acts = torch.from_numpy(np.log(0.02/1.02) * np.ones(task.maxt, dtype = np.float64)))
        # task.plot('constant-reduction')
        
        #probability matcher, starting at 800
        # s = task(acts = torch.from_numpy(get_probability_matcher_actions(800)))
        # task.plot('probability-matcher-800')

        # acts = -1000.0 * np.ones(task.maxt, dtype = np.float64)
        # acts[825] = np.log(0.2)
        # s = task(acts = torch.from_numpy(acts))
        # task.plot('one-time-investment')

        print ('score is', s)
        
