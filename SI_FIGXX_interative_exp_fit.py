#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:31:53 2022

@author: albertsmith
"""


import pyDR
from Fig3A_met_hop_vs_lib import Z2lib,S2_2_libA
from copy import copy
from pyDR.Fitting.fit import model_free
import numpy as np
import matplotlib.pyplot as plt

linear_ex=pyDR.tools.linear_ex

#%% Load the experimental data
exp=pyDR.Project('directHC')
data=exp['NMR']['proc']


#%% Fit first two detectors to methyl hopping/methyl libration
z2lib=Z2lib()
z,sigma,z0=z2lib.fit2z_sigma(data)  #Hopping log(tc),libration stdev, Hopping log(tc) neglecting libration

S2=S2_2_libA.sigma2S2(sigma)  #S2 from libration

#%% Iterative fit of methyl motion and chi motion

data=copy(data)
data.project=None

zchi,Achi=None,None
Smet=1/3
fixA=[(1-exp['NMR']['raw'].S2)-(1-Smet**2)]

for k in range(10):
    print(k)
    if zchi is None:
        z,sigma,_=z2lib.fit2z_sigma(data)
    else:
        data1=copy(data)
        data1.R-=(linear_ex(data.sens.z,data.sens.rhoz,zchi)*Achi).T
        z,sigma,_=z2lib.fit2z_sigma(data1)
    
    S2=S2_2_libA.sigma2S2(sigma)
    diff=copy(data)
    
    Rmet=linear_ex(data.sens.z,data.sens.rhoz,z)*(8/9-1+S2)+\
        np.atleast_2d(data.sens.rhoz[:,0]).repeat(len(z),axis=0).T*(1-S2)
    diff.R-=Rmet.T
    zchi,Achi,err,out=model_free(diff,nz=1,include=range(4),fixA=fixA)
    zchi,Achi=[x.squeeze() for x in [zchi,Achi]]


out=copy(data)
out.R[:]=0
out.R+=(linear_ex(data.sens.z,data.sens.rhoz,[-14 for _ in range(len(S2))])*(1-S2)).T
out.R+=(linear_ex(data.sens.z,data.sens.rhoz,z)*(8/9-1+S2)).T
zchi[-1]=-7.2
out.R+=(linear_ex(data.sens.z,data.sens.rhoz,zchi)*Achi).T

data.project=exp
exp.append_data(out)

data.plot(rho_index=range(4))
out.plot()




sim=pyDR.Project('proj_md/')
out=model_free(sim['OuterHop'],nz=1)
z_no0,z_corr0=[o[0].squeeze() for o in out]

slabel=sim['OuterHop'][0].label

z_no,z_corr=list(),list()
for label in data.label:
    if ',' in label:
        index=[lbl[:3] in [lbl0[:3] for lbl0 in label.split(',')] for lbl in slabel]
    elif label[3:]=='Ala':
        z_no.append(np.nan)
        z_corr.append(np.nan)
        continue
    elif label[3:]=='Ile':
        index=[lbl[:3]==label[:3] and lbl[4:]=='CD' for lbl in slabel]
    elif label[3:] in ['Val','Leu']:
        index=[lbl[:3]==label[:3] for lbl in slabel]
        
    z_no.append(z_no0[index].mean())
    z_corr.append(z_corr0[index].mean())
z_no=np.array(z_no)   
z_corr=np.array(z_corr)

ax=plt.figure().add_subplot(111)
ax.plot(range(len(zchi)),zchi,color='black')
ax.scatter(range(len(zchi)),z_no,marker='x')
ax.scatter(range(len(zchi)),z_corr,marker='o')
ax.set_xticks(range(len(zchi)))
ax.set_xticklabels(data.label,rotation=90)


#%% Second attempt: Iterative fit of methyl motion and chi motion

data=copy(data)
data.project=None

for k in range(5):
    print(k)
    if zchi is None:
        z,sigma,_=z2lib.fit2z_sigma(data)
    else:
        data1=copy(data)
        data1.R-=diff.R
        z,sigma,_=z2lib.fit2z_sigma(data1)
    
    S2=S2_2_libA.sigma2S2(sigma)
    diff=copy(data)
    
    Rmet=linear_ex(data.sens.z,data.sens.rhoz,z)*(8/9-1+S2)+\
        np.atleast_2d(data.sens.rhoz[:,0]).repeat(len(z),axis=0).T*(1-S2)
    diff.R-=Rmet.T
    diff.src_data.S2+=8/9
    diff=diff.opt2dist()


out=copy(data)
out.R[:]=0
out.R+=(linear_ex(data.sens.z,data.sens.rhoz,[-14 for _ in range(len(S2))])*(1-S2)).T
out.R+=(linear_ex(data.sens.z,data.sens.rhoz,z)*(8/9-1+S2)).T
out.R+=diff.R

data.project=exp
exp.append_data(out)

data.plot(rho_index=range(6))
out.plot()


