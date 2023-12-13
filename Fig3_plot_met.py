#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 16:10:54 2022

@author: albertsmith
"""

import pyDR
from met_hop_vs_lib import Z2lib,S2_2_libA
import numpy as np
import matplotlib.pyplot as plt
from pyDR.misc.Averaging import avgData
from pyDR.Fitting.fit import model_free
from pyDR.misc.tools import linear_ex
from copy import copy

#%% Lib vs. Hop
z2lib=Z2lib()
z2lib.Amet=0.93
data=pyDR.Project('Projects/directHC')['NMR']['proc'][0]  #Processed experimental data

#%% Calculate methyl hopping, methyl librational amplitude from data
"""
We do two types of optimization for this calculation. Only this one is shown
in the main text, but we note that the results are almost the same. Interestingly,
results in Figure 4 (distribution of correlation times for non-methyl dynamics)
are much more stable with the latter method. Then, we use this method in
Figure 4
"""
z,sigma,z0=z2lib.fit2z_sigma(data)   #This uses the same function as for simulated data (Fig3_testfit_approach.py)



#%% Plot methyl hopping correlation time, methyl librational amplitude results
fig=plt.figure()
ax=[fig.add_subplot(1,2,k+1) for k in range(2)]
ax[0].plot(z,color='black')
ax[0].set_ylim([-11,-9.5])
ax[0].set_yticks(np.linspace(-11,-9.5,7))
labels=['{:.0f}  ps'.format(10**z*1e12) for k,z in enumerate(ax[0].get_yticks())]
ax[0].set_yticklabels(labels)
ax[0].set_xticks(range(len(z)))
ax[0].set_xticklabels(data.label,rotation=90)
ax[0].set_xlabel('Residue')
ax[0].set_ylabel(r'$\tau_\mathrm{met}$ / ps')
# ax[1].bar(range(len(sigma)),1-S2_2_libA.sigma2S2(sigma))
ax[1].bar(range(len(sigma)),sigma)
ax[1].set_xticks(range(len(z)))
ax[1].set_xticklabels(data.label,rotation=90)
ax[1].set_xlabel('Residue')
# ax[1].set_ylabel(r'$1-S^2_\mathrm{lib.}$')
ax[1].set_ylabel(r'$\sigma$ / $^\circ$')
fig.set_size_inches([9.7,6.6])
fig.tight_layout()

sim=pyDR.Project('Projects/proj_md')['MetHop']  #Methyl hopping data (for two MD simulations)

#%% Find simulated data corresponding to experimental residues
index=list()
for label in data.label:
    i0=np.argwhere([label[:3]==lbl[:3] for lbl in sim[0].label])[:,0]
    if len(i0)==2: #valine/leucine/alanine
        if sim[0].label[i0][0][4:6]==sim[0].label[i0][1][4:6]:  #Valine/Leucine
            index.append(i0)  #Both methyls get averaged together for comparison
        else: #Isoleucine
            index.append(i0[1]) #Only take outermost methyl group for isoleucine
    else:
        index.append(i0[0]) #Alanine
index[9]=[28, 29, 30, 31, 34, 35]  #This is where we have 3 overlapping resonances in the NMR data

#%% Average together simulated data for comparison to experiment
sim=np.sum(avgData(sim,index)) #This averages together data as determined by index

#%% Fits data to single correlation time, plots results for comparison to experiment
out=[model_free(s,nz=1,fixA=8/9) for s in sim]  #Fits each data set (S2=1/9)
zmd=[o[0][0] for o in out]
fit=[o[-1] for o in out]

for zmd0,m in zip(zmd,['o','x']):
    ax[0].scatter(range(len(zmd0)),zmd0,marker=m)
ax[0].legend(('Exper.','methyl corr.','w/o methyl corr.'))


#Compare the detector response for the fit to the original response
sim[0].plot(style='bar')
fit[0].plot()
for a in sim.plot_obj.ax:a.set_ylim([0,1])
sim.plot_obj.show_tc()


#%% Plot in chimera

# First we show the experimental data
minz=z.min()
maxz=z.max()
znorm=(z-maxz)/(minz-maxz)
resids=np.concatenate([[int(lbl[:3])] if ',' not in lbl else [int(l[:3]) for l in lbl.split(',')] for lbl in data.label])
data.select.chimera(x=znorm)
data.project.chimera.command_line(['~show','show /B:'+','.join([str(res) for res in resids]),
                                   'ribbon /B','lighting soft','graphics silhouettes true'])

# data.project.chimera.savefig('methyl_rot.png',options='transparentBackground true')  #Save figure

