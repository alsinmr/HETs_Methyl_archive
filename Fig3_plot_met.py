#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 16:10:54 2022

@author: albertsmith
"""

import pyDR
from met_hop_vs_lib import Z2lib
import numpy as np
import matplotlib.pyplot as plt
from pyDR.misc.Averaging import avgData
from pyDR.Fitting.fit import model_free

#%% Lib vs. Hop
"First we make the plot fitting sigma (methyl libration) to tau_met (methyl hopping)"
z2lib=Z2lib()
z2lib.plot()


data=pyDR.Project('directHC')['NMR']['proc'][0]


"Below is the result of the optimization in Fig4_chi_dist"
i=np.array([32, 89, 43, 63, 55, 52, 85, 59, 79, 72, 49, 76, 65])  #Result of above optimization (zhop=np.linspace(-11.2,-10,101))
zhop=np.linspace(-11.2,-10,101)
z=zhop[i]
sigma=z2lib.z2lib(z)
S2=z2lib.z2S2(z)


# z,sigma,z0=z2lib.fit2z_sigma(data)  #This is an alternative way to get z. Slightly less accurate



fig=plt.figure()
ax=[fig.add_subplot(1,2,k+1) for k in range(2)]
ax[0].plot(z,color='black')
# ax[0].plot(z0)
ax[0].set_ylim([-11,-9.5])
ax[0].set_yticks(np.linspace(-11,-9.5,7))
labels=['{:.0f}  ps'.format(10**z*1e12) for k,z in enumerate(ax[0].get_yticks())]
ax[0].set_yticklabels(labels)
# ax[0].legend(('Libr. Corrected','Uncorrected'))
ax[0].set_xticks(range(len(z)))
ax[0].set_xticklabels(data.label,rotation=90)
ax[0].set_xlabel('Residue')
ax[0].set_ylabel(r'$\tau_\mathrm{met}$ / ps')
ax[1].bar(range(len(sigma)),sigma)
ax[1].set_xticks(range(len(z)))
ax[1].set_xticklabels(data.label,rotation=90)
ax[1].set_xlabel('Residue')
ax[1].set_ylabel(r'$\sigma$ / $^\circ$')
fig.set_size_inches([9.7,6.6])
fig.tight_layout()

sim=pyDR.Project('proj_exp')['MetHop']

index=list()
for label in data.label:
    i0=np.argwhere([label[:3]==lbl[:3] for lbl in sim[0].label])[:,0]
    if len(i0)==2:
        if sim[0].label[i0][0][4:6]==sim[0].label[i0][1][4:6]:
            index.append(i0)
        else:
            index.append(i0[1])
    else:
        index.append(i0[0])
index[9]=[28, 29, 30, 31, 34, 35]

sim=np.sum(avgData(sim,index))

zmd=[model_free(s,nz=1)[0][0] for s in sim]

for zmd0,m in zip(zmd,['o','x']):
    ax[0].scatter(range(len(zmd0)),zmd0,marker=m)
ax[0].legend(('Exper.','methyl corr.','w/o methyl corr.'))


#%% Plot in chimera

minz=z.min()
maxz=z.max()
znorm=(z-maxz)/(minz-maxz)
resids=np.concatenate([[int(lbl[:3])] if ',' not in lbl else [int(l[:3]) for l in lbl.split(',')] for lbl in data.label])
data.select.chimera(x=znorm)
data.project.chimera.command_line(['~show','show /B:'+','.join([str(res) for res in resids]),
                                   'ribbon /B','lighting soft','graphics silhouettes true'])



znorm=(zmd[0]-maxz)/(minz-maxz)
sim[0].select.chimera(x=znorm)
sim.chimera.command_line(['~show','show /B:'+','.join([str(res) for res in resids]),
                                   'ribbon /B','lighting soft','graphics silhouettes true'])


#%% Here we verify that this approach works 