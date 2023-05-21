#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:37:04 2022

@author: albertsmith
"""
import pyDR
from met_hop_vs_lib import Z2lib
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from Fig4_plot_chi_pop import load_md


linear_ex=pyDR.tools.linear_ex


S2hop=0.09667605  #Typically 1/9
# S2hop=1/9
#%% Get list of residues to test
resids=np.concatenate([[int(l0[:3]) for l0 in lbl.split(',')] if ',' in lbl else [int(lbl[:3])] \
                       for lbl in pyDR.Project('directHC')['NMR'][0].label])

proj=pyDR.Project('proj_exp')['HETs_MET_4pw_SegB']

data=proj['Direct']['opt_fit'][0]




#%% Get S2 for the trajectory
def calcS2(pop):
    return (pop**2).sum()-2/3*(pop[0]*pop[1]+pop[0]*pop[1]+pop[1]*pop[2])

_,md_corr=load_md()
S2chi=list()
for lbl in data.label:
    if ',' in lbl:
        S2chi.append(np.mean([calcS2(md_corr[int(l0[:3])]['pop_outer']) for l0 in lbl.split(',')]))
    elif int(lbl[:3]) in md_corr:
        S2chi.append(calcS2(md_corr[int(lbl[:3])]['pop_outer']))
    else:
        S2chi.append(1)
np.array(S2chi)


#%% Fit using same approach as for experimental data
z2lib=Z2lib()
z,sigma,z0=z2lib.fit2z_sigma(data)  #Hopping log(tc),libration stdev, Hopping log(tc) neglecting libration

data.project=None
error=list()
zhop=np.linspace(-11.2,-10,101)
# zhop=np.linspace(-11.2,-9.52,36)

# for zhop0 in zhop:    
#     print(zhop0)
#     S2lib=z2lib.z2S2(zhop0)
#     Rmet=linear_ex(data.sens.z,data.sens.rhoz,zhop0)*(S2lib-S2hop)+\
#         data.sens.rhoz[:,0]*(1-S2lib)
#     diff=copy(data)
#     diff.R-=Rmet
#     diff.src_data=copy(diff)  #Just a place holder for src_data, but we need to include S2
#     diff.src_data.S2=np.array(S2chi)
#     diff=diff.opt2dist()
#     diff.R+=Rmet
    
#     error.append(np.abs(diff.R-data.R).sum(axis=1))

# data.project=proj._parent
# error=np.array(error)
# i=np.argmin(error,axis=0)

# i=np.array([32, 30, 74, 57, 14, 43, 26, 60, 32, 56, 81, 39, 40, 61, 67, 45, 49,
#         49, 43, 60, 66, 77, 58, 56, 59, 36, 54, 18, 60, 62, 31, 44, 43, 44,
#         34, 29, 59, 76, 55, 32])  #Result of above optimization, zhop=np.linspace(-11.2,-10,101)

# z=zhop[i]



S2lib=z2lib.z2S2(z)

Rmet=linear_ex(data.sens.z,data.sens.rhoz,z)*(S2lib-S2hop)+\
    np.atleast_2d(data.sens.rhoz[:,0]).repeat(len(z),axis=0).T*(1-S2lib)
diff=copy(data)
diff.R-=Rmet.T
diff.src_data=copy(diff)  #Just a place holder for src_data, but we need to include S2
diff.src_data.S2=np.array(S2chi)
diff=diff.opt2dist()    #Optimize dectector responses (consistent with distribution)
diff.R*=9  #Multiply by 9 to counter methyl scaling


fr=proj['MetHop']['opt_fit'][0]
index=[int(lbl[:3]) in resids for lbl in fr.label]



fr.plot(index=index,rho_index=range(4))
diff.plot()

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

