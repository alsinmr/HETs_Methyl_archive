#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:12:43 2022

@author: albertsmith
"""

import sys
sys.path.append('/Users/albertsmith/Documents/GitHub/')
import pyDR
import matplotlib.pyplot as plt
import numpy as np


trajs=['HETs_3pw','HETs_4pw','HETs_MET_3pw','HETs_MET_4pw',
        'HETs_MET_4pw_2kj3','HETs_MET_4pw_5chain']

titles=['3pw','4pw','3pw+methyl corr.','4pw+methyl corr.',
        '4pw+methyl corr.\n(2kj3)','4pw+methyl corr.\n(5chains)']
nd=[4,3]

projHC=pyDR.Project('directHC')
projHN=pyDR.Project('directHN')

#%% Bar plot of the total errors
chiHC=list()
chiHN=list()
for chi,nd0,proj in zip([chiHC,chiHN],nd,[projHC,projHN]):
    R,Rstd=[getattr(proj['NMR']['proc'],k)[:,:nd0] for k in ['R','Rstd']]
    for traj,title in zip(trajs,titles):
        chi.append((R-proj['opt_fit'][traj].R[:,:nd0])**2/Rstd**2)

chiHC=np.array(chiHC)
chiHN=np.array(chiHN)    

clrs=plt.get_cmap('tab10')

ax=plt.figure().add_subplot(111)
for k,x in enumerate(range(len(trajs),0,-1)):
    ax.barh(x,chiHC[k].sum()+chiHN[k].sum(),color=clrs(k),hatch='x',edgecolor='black')
    ax.barh(x,chiHC[k].sum(),color=clrs(k),hatch='oo',edgecolor='black')
    
ax.set_yticks(range(1,len(trajs)+1))
ax.set_yticklabels(titles[::-1])
ax.set_xlabel(r'$\chi^2$')
ax.figure.tight_layout()

#%% Bar plot of 4pw with and without methyl correction vs experiments (Methyl relaxation)
projHC['NMR']['proc'].plot(color='black',errorbars=True)
(projHC['opt_fit']['HETs_4pw']+projHC['opt_fit']['HETs_MET_4pw']).plot(style='bar')

