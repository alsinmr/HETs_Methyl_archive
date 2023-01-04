#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:33:57 2022

@author: albertsmith
"""


import pyDR
from Fig3A_met_hop_vs_lib import Z2lib,S2_2_libA
from copy import copy
from pyDR.misc.Averaging import avgData
from pyDR.Fitting.fit import model_free
import numpy as np
import matplotlib.pyplot as plt

linear_ex=pyDR.tools.linear_ex

#%% Load the experimental data
exp=pyDR.Project('directHC')
data=exp['NMR']['proc'][0]


#%% Fit first two detectors to methyl hopping/methyl libration
z2lib=Z2lib()
data.project=None  #Detach from project for fitting
z,sigma,z0=z2lib.fit2z_sigma(data)  #Hopping log(tc),libration stdev, Hopping log(tc) neglecting libration


error=list()
zhop=np.linspace(-11.2,-10,26)
# zhop=np.linspace(-11.2,-9.52,36)
for zhop0 in zhop:    
    print(zhop0)
    S2=z2lib.z2S2(zhop0)
    Rmet=linear_ex(data.sens.z,data.sens.rhoz,zhop0)*(8/9-1+S2)+\
        data.sens.rhoz[:,0]*(1-S2)
    diff=copy(data)
    diff.R-=Rmet
    diff.src_data.S2+=8/9
    diff=diff.opt2dist()
    diff.src_data.S2-=8/9
    diff.R+=Rmet
    
    error.append(np.abs(diff.R-data.R).sum(axis=1))

error=np.array(error)
i=np.argmin(error,axis=0)

# i=array([ 8, 22, 11, 15, 14, 13, 21, 14, 20, 18, 12, 19, 16])  #Result of above optimization (zhop=np.linspace(-11.2,-10,26))
z=zhop[i]


# z,sigma,z0=z2lib.fit2z_sigma(data)  #Hopping log(tc),libration stdev, Hopping log(tc) neglecting libration



S2=z2lib.z2S2(z)

out=copy(data)
Rmet=linear_ex(data.sens.z,data.sens.rhoz,z)*(8/9-1+S2)+\
    np.atleast_2d(data.sens.rhoz[:,0]).repeat(len(z),axis=0).T*(1-S2)
diff=copy(data)
diff.R-=Rmet.T
diff.src_data.S2+=8/9
diff=diff.opt2dist()
diff.src_data.S2-=8/9
diff.R*=9  #Multiply by 9 to counter methyl scaling


out.R[:]=0

out.R+=Rmet.T   #Contribution from methyl hop/lib
out.R+=diff.R/9     #Contrbution from other motion

data.project=exp
exp.append_data(out)


sim=pyDR.Project('proj_exp/')['OuterHop']
sim_r0e=pyDR.Project('proj_exp_r0e')['OuterHop']
slabel=sim[0].label


index=list()
for label in data.label:
    if ',' in label:
        lbls=[lbl[:3] for lbl in label.split(',')]
        index.append(np.argwhere([sl[:3] in lbls for sl in slabel])[:,0])
    elif label[3:] in ['Val','Leu','Ala']:
        index.append(np.argwhere([sl[:3]==label[:3] for sl in slabel])[:,0])
    else:
        index.append(np.argwhere([sl[:3]==label[:3] and sl[4:]=='CD' for sl in slabel])[:,0])

sim=np.sum(avgData(sim,index))
sim_r0e=np.sum(avgData(sim_r0e,index))


index1=['Ala' not in lbl for lbl in diff.label]
plt_obj=diff.plot(color='black',rho_index=range(4),index=index1)
hatch=[None,'///']
w=.45
for k,(s,h) in enumerate(zip(sim_r0e,hatch)):
    plt_obj.ax[0].bar(np.arange(sum(index))-w/2+k*w,s.R[index,0],color='lightgrey',hatch=h,width=w)
plt_obj.ax_sens.plot(sim_r0e[0].sens.z,sim_r0e[0].sens.rhoz[0],color='black',linestyle='-')
for s in sim:plt_obj.append_data(s,errorbars=False,style='bar')

for a in plt_obj.ax:a.set_ylim([0,1])
for a in plt_obj.ax:a.plot(range(sum(index)),(1/9-data.src_data.S2[index])*9,linestyle=':',color='grey')


sim1=pyDR.Project('Frames')['Avg_chi_hop>side_chain_chi']['HETs_MET_4pw']['no_opt'][0]
sim1.detect.r_auto(12)
sim1=sim1.fit().opt2dist(rhoz_cleanup=True)

# sim1=pyDR.Project('proj_md')['OuterHop'][1]

from Fig3C_plot_chi_pop import load_md
_,md_corr=load_md()

def plot_residue(resid,ax=None):
    if ax is None:ax=plt.figure().add_subplot(111)
    ax.set_xlim([-12,-6])
    
    i=[str(resid)==lbl[:3] for lbl in diff.label]
    
    label=diff.label[i][0]
    
    #Plot simulation
    if label[3:]!='Ala':
        z=[-11.45,*sim1.sens.info['z0'][1:-2]]     
        index=[lbl[:3] in [l0[:3] for l0 in label.split(',')] for lbl in sim1.label]
        R=sim1.R[index,:-2].mean(axis=0)
        w=.15
        for z0,R0 in zip(z,R):ax.bar(z0,R0,width=w,facecolor='black',edgecolor='grey')
        pop=md_corr[int(label[:3])]['pop_outer']
        S2=(pop**2).sum()-2/3*(pop[0]*pop[1]+pop[0]*pop[2]+pop[1]*pop[2])
        ax.plot(ax.get_xlim(),(1-S2)*np.ones(2),color='black',linestyle='--')
    
    #Plot experiment
    z=[-10.8,*diff.sens.info['z0'][1:4],-7.9]
    R=diff.R[i,[0,1,2,3,0,4,5]]
    clr=plt.get_cmap('tab10')([0,1,2,3,0,4,5])
    if R[1]>R[3]:clr[4]=plt.get_cmap('tab10')(7)
    else:clr[0]=plt.get_cmap('tab10')(7)
    clr[:,-1]=.75
    w=.3
    for z0,R0,clr0 in zip(z,R,clr):ax.bar(z0,R0,width=w,facecolor=clr0,edgecolor='black')

    
    
    ax.plot(ax.get_xlim(),(1-diff.src_data.S2[i]-8/9)*np.ones(2)*9,color='black',linestyle=':')
    
    
    ax.set_ylim([0,.9])
    
    ax.set_xticks(range(-12,-6))
    ax.set_xticklabels([None if not(z%2) else ('{:.0f} ps'.format(10**z*1e12) if z<-9 else '{:.0f} ns'.format(10**z*1e9)) for z in range(-12,-6)]) 
    ax.set_xlabel(r'$\tau_c$')
    ax.text(-11.5,0.8,label)
    

resids=list()
for lbl in diff.label:
    if lbl[3:]!='Ala':resids.append(int(lbl[:3]))

# resids=[int(lbl[:3]) for lbl in diff.label]

fig=plt.figure()
ax=[fig.add_subplot(3,3,k+1) for k in range(9)] if len(resids)<=9 else [fig.add_subplot(4,4,k+1) for k in range(13)]

for r,a in zip(resids,ax):plot_residue(r,ax=a)
fig.set_size_inches([12.5,9])
fig.tight_layout()




    
    

