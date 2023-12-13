#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:12:43 2022

@author: albertsmith
"""

import pyDR
import matplotlib.pyplot as plt
import numpy as np


# Names of each trajectory in the project directories
trajs=['HETs_3pw','HETs_4pw','HETs_MET_tip3p_870ns','HETs_MET_3pw','HETs_MET_4pw',
        'HETs_MET_4pw_2kj3','HETs_MET_4pw_5chain']

# Titles for the bar plots
titles=['SPC/E','TIP4P','TIP3P+methyl corr.','SPC/E+methyl corr.','TIP4P+methyl corr.',
        'TIP4P+methyl corr.\n(2kj3)','TIP4P+methyl corr.\n(5chains)']

# Number of detectors to sum over (4 for methyl data, 3 for backbone data)
nd=[4,3]

projHC=pyDR.Project('Projects/directHC')  #Methyl data project (exper+sim)
projHN=pyDR.Project('Projects/directHN')  #Backbone data project (exper+sim)

#%% Bar plot of the total errors
chiHC=list()  #Collect the total errors in these two lists
chiHN=list()
for chi,nd0,proj in zip([chiHC,chiHN],nd,[projHC,projHN]):
    R,Rstd=[getattr(proj['NMR']['proc'],k)[:,:nd0] for k in ['R','Rstd']]
    for traj,title in zip(trajs,titles):
        chi.append((R-proj['opt_fit'][traj].R[:,:nd0])**2/Rstd**2)

# Convert lists to numpy arrays
chiHC=np.array(chiHC)
chiHN=np.array(chiHN)    

clrs=plt.get_cmap('tab10')

# Plot the errors
ax=plt.figure().add_subplot(111)
for k,x in enumerate(range(len(trajs),0,-1)):
    ax.barh(x,chiHC[k].sum()+chiHN[k].sum(),color=clrs(7),hatch='x',edgecolor='black')
    ax.barh(x,chiHC[k].sum(),color=clrs(7),hatch='oo',edgecolor='black')
    
ax.set_yticks(range(1,len(trajs)+1))
ax.set_yticklabels(titles[::-1])
ax.set_xlabel(r'$\chi^2$')
ax.figure.tight_layout()

#%% Bar plot of 4pw with and without methyl correction vs experiments (Methyl relaxation)
projHC['NMR']['proc'].plot(color='black',errorbars=True)
(projHC['opt_fit']['HETs_4pw']+projHC['opt_fit']['HETs_MET_4pw']).plot(style='bar')

#%% Bar plot of the error for each of four detectors
nd=4
chiMET=((projHC['opt_fit']['HETs_MET_4pw'].R[:,:nd]-projHC['NMR']['proc'].R[:,:nd])**2/projHC['NMR']['proc'].Rstd[:,:nd]**2).sum(0)
chiNOMET=((projHC['opt_fit']['HETs_4pw'].R[:,:nd]-projHC['NMR']['proc'].R[:,:nd])**2/projHC['NMR']['proc'].Rstd[:,:nd]**2).sum(0)

fig,ax=plt.subplots(4,1,sharey=True)
ax=ax.flatten()
cmap=plt.get_cmap('tab10')
for k,(a,chi0,chi1) in enumerate(zip(ax,chiNOMET,chiMET)):
    a.bar([0],[np.log10(chi0)],width=1,color=cmap(k))
    a.bar([1],[np.log10(chi1)],width=1,color=cmap(k),hatch='//')

    
