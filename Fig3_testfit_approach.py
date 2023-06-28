#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:37:04 2022

@author: albertsmith
"""
import pyDR
from met_hop_vs_lib import Z2lib
import numpy as np
import matplotlib.pyplot as plt



linear_ex=pyDR.tools.linear_ex


S2hop=0.09667605  #Typically 1/9
# S2hop=1/9
#%% Get list of residues to test
#List of residues
resids=np.concatenate([[int(l0[:3]) for l0 in lbl.split(',')] if ',' in lbl else [int(lbl[:3])] \
                       for lbl in pyDR.Project('Projects/directHC')['NMR'][0].label])

#Subproject containing the best force field (Methyl correction with 4 point water)    
proj=pyDR.Project('Projects/proj_exp')['HETs_MET_4pw_SegB']

#Data for total motion
data=proj['Direct']['opt_fit'][0]



#%% Fit using same approach as for experimental data
z2lib=Z2lib()  #Object for various operations dealing with methyl hopping and libration
"First we make the plot fitting sigma (methyl libration) to tau_met (methyl hopping)"
z2lib.plot()  #Shows the MD-derived relationship between methyl libr. amplitude and methyl hopping rate
z,sigma,z0=z2lib.fit2z_sigma(data)  #Hopping log(tc),libration stdev, Hopping log(tc) neglecting libration



ax=plt.figure().add_subplot(111)
ax.plot(z2lib.z)
ax.plot(z)
ax.set_xticks(range(len(z)))
ax.set_xticklabels(data.label,rotation=90)
ax.figure.tight_layout()

ax=plt.figure().add_subplot(111)
ax.plot(z2lib.sigma)
ax.plot(sigma)
ax.set_xticks(range(len(z)))
ax.set_xticklabels(data.label,rotation=90)
ax.figure.tight_layout()

