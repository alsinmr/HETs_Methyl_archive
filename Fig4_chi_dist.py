#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:33:57 2022

@author: albertsmith
"""


import pyDR
from met_hop_vs_lib import Z2lib
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

linear_ex=pyDR.tools.linear_ex

#%% Load the experimental data
exp=pyDR.Project('Projects/directHC')
data=exp['NMR']['proc'][0]

S2hop=1/10
#%% Fit first two detectors to methyl hopping/methyl libration
z2lib=Z2lib()
# Amet=0.93
Amet=1-S2hop
z2lib.Amet=Amet

z,sigma,z0=z2lib.fit2z_sigma(data,n=2)  #Hopping log(tc),libration stdev, Hopping log(tc) neglecting libration

S2lib=z2lib.z2S2(z)  # Calculates librational component to S2

Rmet=linear_ex(data.sens.z,data.sens.rhoz,z)*(Amet-1+S2lib)+\
    np.atleast_2d(data.sens.rhoz[:,0]).repeat(len(z),axis=0).T*(1-S2lib)
diff=copy(data)         #We'll modify diff to factor out influence of methyl motion (hop+lib)
diff.R-=Rmet.T          #Subtract away methyl contribution
diff.src_data.S2/=S2hop   #Add Amet back to S2
# diff.src_data.S2=1-(1-diff.src_data.S2)/(1-Amet)   #Scale up the source data S2 (remove methyl scaling)

diff.plot(style='bar')
diff.R/=S2hop
diff=diff.opt2dist()    #Optimize dectector responses (consistent with distribution)




# This is a back-calculated data set of describing the total motion
out=copy(data)
out.R=Rmet.T          #Contribution from methyl hop/lib
out.R+=diff.R/9     #Contrbution from other motion

exp.append_data(out)

exp['proc']['NMR'].plot()  #Compare original data NMR processed data set to back-calculated data set


#%% Now analyze simulated data with 9 detectors
n=9
norm='I'

sim=pyDR.Project('Projects/methyl')['Avg_AvOb_methylCC>LF']['no_opt'][0]  #All motion except methyl dynamics
sim.detect.r_auto(n,Normalization=norm)      #Set up detectors
sim=sim.fit().opt2dist(rhoz_cleanup=True)   #Detector analysis+optimization

sim1=pyDR.Project('Projects/frames_s')['Avg_AvOb_chi_hop>side_chain_chi']['HETs_MET_4pw_SegB']['no_opt'][0] #Outermost hop
# sim1.detect.r_target(target)
sim1.detect.r_auto(n,Normalization=norm)
sim1=sim1.fit().opt2dist(rhoz_cleanup=True)

# sim=pyDR.Project('proj_md')['OuterHop'][1]

from Fig4_plot_chi_pop import load_md
_,md_corr=load_md()

def plot_residue(resid,ax=None):
    if ax is None:ax=plt.figure().add_subplot(111)
    
    
    i=[str(resid)==lbl[:3] for lbl in diff.label]
    
    label=diff.label[i][0]
    
    #Plot simulation
    if label[3:]!='Ala':
        ii=-2
        # hdl=sim.sens.plot_rhoz(index=range(sim.sens.rhoz.shape[0]+ii),color='lightgrey',ax=ax)
        z=[-11.45,*sim.sens.info['z0'][1:]]     
        if label[3:]=='Ile':
            index=[lbl[:3]==label[:3] and lbl[4:6]=='CD' for lbl in sim.label]
        elif ',' in label:
            index=[lbl[:3] in [lbl0[:3] for lbl0 in label.split(',')] for lbl in sim.label]
        else:
            index=[lbl[:3]==label[:3] for lbl in sim.label]
        R=sim.R[index].mean(axis=0)
        R1=sim1.R[index].mean(axis=0)
        w=.15
        
        
        coords=np.concatenate(([[*z[:ii],z[ii-1],z[0]]],[[*R[:ii],0,0]]),axis=0).T
        polygons = [];
        polygons.append(Polygon(coords))
        p = PatchCollection(polygons, alpha=0.7)
        p.set_facecolor('lightgrey')
        ax.add_collection(p)
        coords=np.concatenate(([[*z[:ii],z[ii-1],z[0]]],[[*R1[:ii],0,0]]),axis=0).T
        polygons = [];
        polygons.append(Polygon(coords))
        p = PatchCollection(polygons, alpha=0.7)
        p.set_facecolor('darkgrey')
        ax.add_collection(p)
        # for z0,R0 in zip(z,R):ax.bar(z0,R0,width=w,facecolor='black',edgecolor='grey')
        # for z0,R0 in zip(z,R1):ax.bar(z0,R0,width=w,facecolor='grey',edgecolor='grey')
        
        
        
        # pop=md_corr[int(label[:3])]['pop_outer']
        # S2=(pop**2).sum()-2/3*(pop[0]*pop[1]+pop[0]*pop[2]+pop[1]*pop[2])
        # ax.plot(ax.get_xlim(),(1-S2)*np.ones(2),color='black',linestyle='--')
        
        ax.set_xlim([-12,z[ii-1]])
    else:
        ax.set_xlim([-12,-7])
    
    #Plot experiment
    #These four lines switch which bar is blue/grey
    # z=[-10.8,*diff.sens.info['z0'][1:4],-7.9]
    
    # R=diff.R[i,[0,1,2,3,0,4,5]]
    # clr=plt.get_cmap('tab10')([0,1,2,3,0,4,5])
    # if R[1]>R[3]:clr[4]=plt.get_cmap('tab10')(7)
    # else:clr[0]=plt.get_cmap('tab10')(7)
    
    #New plotting strategy
    z=[-10.8,*diff.sens.info['z0'][1:4],-7.9]    
    R=diff.R[i,[0,1,2,3,0,4,5]]
    R[[0,4]]/=2
    clr=plt.get_cmap('tab10')([0,1,2,3,0,4,5])
    
    
    

    
    clr[:,-1]=.75
    w=.3
    for z0,R0,clr0 in zip(z,R,clr):ax.bar(z0,R0,width=w,facecolor=clr0,edgecolor='black')

    
    
    ax.plot(ax.get_xlim(),(1-diff.src_data.S2[i]-8/9)*np.ones(2)*9,color='black',linestyle=':')
    
    
    ax.set_ylim([0,.9])
    
    ax.set_xticks(range(-12,-6))
    ax.set_xticklabels([None if not(z%2) else ('{:.0f} ps'.format(10**z*1e12) if z<-9 else '{:.0f} ns'.format(10**z*1e9)) for z in range(-12,-6)]) 
    ax.set_xlabel(r'$\tau_c$')
    ax.text(-11.5,0.8,label)
    ax.hlines(1-diff.src_data.S2[i],*ax.get_xlim(),color='black',linestyle=':')
    

resids=list()
for lbl in diff.label:
    if lbl[3:]!='Ala':resids.append(int(lbl[:3]))

# resids=[int(lbl[:3]) for lbl in diff.label]



fig=plt.figure()
ax=[fig.add_subplot(3,3,k+1) for k in range(9)] if len(resids)<=9 else [fig.add_subplot(4,4,k+1) for k in range(13)]

for r,a in zip(resids,ax):plot_residue(r,ax=a)
fig.set_size_inches([12.5,9])
fig.tight_layout()



#%% Chimera plots
exp.chimera.saved_commands=['~show','show /B:'+','.join([str(res) for res in resids]),'ribbon /B',
              'turn y -15','turn z 20','turn y -90','turn x -35','turn z 5',
              'view','zoom 1.4','lighting soft','graphics silhouettes true']
for k in range(1,4):
    exp.chimera.close()
    diff.chimera(rho_index=k,scaling=1.5)
    exp.chimera.savefig(f'exp_nomet_rho{k}.png',
                        options='transparentBackground true')

    
    

