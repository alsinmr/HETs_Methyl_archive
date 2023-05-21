#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 13:42:24 2023

@author: albertsmith
"""

import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt
import pyDR    #This adds some functionality to the matplotlib axes
import os


id0=np.load('indexfile.npy',allow_pickle=True).item()

#%% Rearrange the index
index=dict()

for key,value in id0.items():
    if np.any(['_A' in k for k in value.keys()]):
        index[key+'_A']={'MET':value[key+'_A'],'chi1':value['chi1']}
        index[key+'_B']={'MET':value[key+'_B'],'chi1':value['chi1']}    #For ILE, this is the delta carbon
        if 'chi2' in value.keys():
            index[key+'_A']['chi2']=value['chi2']
            if 'LEU' in key:
                index[key+'_B']['chi2']=value['chi2']
    else:
        index[key]={'MET':value[key]}
        if 'chi1' in value.keys():
            index[key]['chi1']=value['chi1']
        if 'chi2' in value.keys():
            index[key]['chi2']=value['chi2']
            

#%% Load data
met=np.mod(np.load('dih_MET_4pw_10μs.npy'),360)
chi1=np.mod(np.load('dih_chi1_MET_4pw_10μs.npy'),360)
chi2=np.mod(np.load('dih_chi2_MET_4pw_10μs.npy'),360)


#%% Determine state of each angle (0, 1, or 2)
ref0=np.mod(met,120).mean(1)
ref_met=np.concatenate(([ref0],[ref0+120],[ref0+240]),axis=0)
ref0=np.mod(chi1,120).mean(1)
ref_chi1=np.concatenate(([ref0],[ref0+120],[ref0+240]),axis=0)
ref0=np.mod(chi2,120).mean(1)
ref_chi2=np.concatenate(([ref0],[ref0+120],[ref0+240]),axis=0)

state_met=np.argmin([np.abs(met.T-ref) for ref in ref_met],axis=0)
state_chi1=np.argmin([np.abs(chi1.T-ref) for ref in ref_chi1],axis=0)
state_chi2=np.argmin([np.abs(chi2.T-ref) for ref in ref_chi2],axis=0)

state_chi12=np.zeros(state_chi2.shape)
for key,value in index.items():
    if 'chi2' in value:
        state_chi12[:,value['chi2']]=state_chi1[:,value['chi1']]+3*state_chi2[:,value['chi2']]


#%% Determine hop times
hop_met=np.diff(state_met,axis=0).astype(bool)
hop_chi1=np.diff(state_chi1,axis=0).astype(bool)
hop_chi2=np.diff(state_chi2,axis=0).astype(bool)

#%% Plot probabilities for methyl groups with chi1 and chi2 (Leu, Ile B)

directory='/Users/albertsmith/Documents/Dynamics/HETs_Methyl_Loquet/Paper/Figures/figure_parts/Pop_vs_hop/'

def disp_count(h,s,ax):
    for k in range(3):
        pct=(s==k).sum()/len(s)*100
        string=f'{pct:.0f}'
        if string=='0':string=f'{pct:.0e}'
        ax.text(k,np.mean(ax.get_ylim()),string,ha='center')

counter=0
fig,ax=plt.subplots(12,4)
fig.set_size_inches([9.5,10.5])

for key,value in index.items():
    if 'chi2' in value:
        ax0=ax[counter]
        counter+=1
        
        hm,hc1,hc2=hop_met.T[value['MET']],hop_chi1.T[value['chi1']],hop_chi2.T[value['chi2']]  #Select hops for this residue
        sm,sc1,sc2=state_met.T[value['MET']],state_chi1.T[value['chi1']],state_chi2.T[value['chi2']]  #Select hops for this residue
        sm,sc1,sc2=sm[:-1],sc1[:-1],sc2[:-1]

        Rmet_v_chi1=[hm[sc1==k].sum()/(sc1==k).sum()/.005 for k in range(3)]
        Rmet_v_chi2=[hm[sc2==k].sum()/(sc2==k).sum()/.005 for k in range(3)]
        Rchi1_v_chi2=[hc1[sc2==k].sum()/(sc2==k).sum()/.005 for k in range(3)]
        Rchi2_v_chi1=[hc2[sc1==k].sum()/(sc1==k).sum()/.005 for k in range(3)]
        
        ax0[0].bar(range(3),Rmet_v_chi1)
        disp_count(hm,sc1,ax0[0])
        ax0[1].bar(range(3),Rmet_v_chi2)
        disp_count(hm,sc2,ax0[1])
        ax0[2].bar(range(3),Rchi1_v_chi2)
        disp_count(hc1, sc2, ax0[2])
        ax0[3].bar(range(3),Rchi2_v_chi1)
        disp_count(hc2,sc1,ax0[3])
        for a in ax0:
            a.set_xticks([0,1,2])
        ax0[0].text(0,ax0[0].get_ylim()[1]*.8,key[4:7]+key[:3],fontsize=8)
        if not(ax0[0].get_subplotspec().is_last_row()):
            for a in ax0:
                a.set_xticklabels('')
ax[0][0].set_title(r'Met v. $\chi_1$',fontsize=10)
ax[0][1].set_title(r'Met v. $\chi_2$',fontsize=10)
ax[0][2].set_title(r'$\chi_1$ v. $\chi_2$',fontsize=10)
ax[0][3].set_title(r'$\chi_2$ v. $\chi_1$',fontsize=10)
fig.tight_layout()        


#%% Binned rate constants for methyl hopping
n=10000
nr=hop_met.shape[1]
N=hop_met.shape[0]//n
dt=.005*n
t=np.arange(N)*dt/1e3

Rmet=np.zeros([N,hop_met.shape[1]])    #Rates of hopping
Rchi1=np.zeros([N,hop_chi1.shape[1]])
Rchi2=np.zeros([N,hop_chi2.shape[1]])

frac_chi1=np.zeros([3,N,state_chi1.shape[1]])
frac_chi2=np.zeros([3,N,state_chi2.shape[1]])
frac_chi12=np.zeros([9,N,state_chi12.shape[1]])

for k in range(n):
    Rmet+=hop_met[k::n]/dt
    Rchi1+=hop_chi1[k::n]/dt
    Rchi2+=hop_chi2[k::n]/dt
    for m in range(3):
        frac_chi1[m]+=(state_chi1[k:-1:n]==m)/n
        frac_chi2[m]+=(state_chi2[k:-1:n]==m)/n
    for m in range(9):
        frac_chi12[m]+=(state_chi12[k:-1:n]==m)/n

#%% Create a selection object for data viewing
# sel=pyDR.MolSelect(topo='/Volumes/My Book/HETs/HETs_SegB.pdb')  #We'll display correlation coefficients from this object

# for key,value in index.items():

    
def set_axis(Rmax,ax):
    count=0
    while Rmax>10:
        count+=1
        Rmax/=10
    if Rmax<2:
        yl=np.ceil(Rmax*10)
        Rmax*=10**count
        yl=[0,yl*10**(count-1)]
    else:
        yl=[0,np.ceil(Rmax)*10**count]
        Rmax*=10**count
    ax.set_ylim(yl)
    return yl
    

#%% Hop probability for Alanine

fig,ax=plt.subplots(2,3)
fig.set_size_inches([9.5,5.5])
ax=ax.flatten()

counter=0
for key,value in index.items():
    if 'ALA' in key:
        ax0=ax[counter]
        counter+=1
        
        R=Rmet.T[value['MET']]
        
        yl=set_axis(R.max(),ax0)
        
        ax0.plot(t,R,color='black')
        
        if ax0.is_first_col():
            ax0.set_ylabel(r'$R_{hop}$ / ns$^{-1}$')
        # else:
        #     ax0.set_yticklabels('')
        if ax0.is_last_row():
            ax0.set_xlabel(r'$t$ / $\mu$s')
        else:
            ax0.set_xticklabels('')
            
        ax0.set_title(key[4:7]+key[:3])
        ax0.set_xlim([0,t[-1]])

fig.tight_layout()
fig.savefig(os.path.join(directory,'ala_hop.pdf'))

#%% Hop probability vs. state : ILE and LEU

core=[231,239,241,264,267,275,277]
CCcore=list()
CCouter=list()

cmap0=plt.get_cmap('tab20c')
cmap=[cmap0(0),cmap0(1),cmap0(2),cmap0(4),cmap0(5),cmap0(6),cmap0(8),cmap0(9),cmap0(10)]

fig,ax=plt.subplots(4,3)
fig.set_size_inches([9.5,10.5])
ax=ax.flatten()

counter=0
for key,value in index.items():
    if 'chi2' in value:
        ax0=ax[counter]
        counter+=1
        
        R=Rmet.T[value['MET']]
        fc12=frac_chi12.T[value['chi2']].T
        
        a=np.linalg.pinv(fc12.T)@R
        Rc=a@fc12
        
        cc=((R-R.mean())*(Rc-Rc.mean())).mean()/np.std(R)/np.std(Rc)
        
        yl=set_axis(R.max(),ax0)
        
        for k,color in enumerate(cmap):
            ax0.fill_between(t,fc12[:k].sum(0)*yl[1],fc12[:k+1].sum(0)*yl[1],color=color)
        ax0.plot(t,R,color='black')
        ax0.plot(t,Rc,color='white',linestyle='--',linewidth=2)
        
        if ax0.is_first_col():
            ax0.set_ylabel(r'$R_{hop}$ / ns$^{-1}$')
        # else:
        #     ax0.set_yticklabels('')
        if ax0.is_last_row():
            ax0.set_xlabel(r'$t$ / $\mu$s')
        else:
            ax0.set_xticklabels('')
            
        ax0.set_title(key[4:7]+key[:3])
        ax0.set_xlim([0,t[-1]])
        
        ax0.text(.5,yl[1]*.8,f'cc = {cc:.2f}',color='white' if int(key[4:7]) in core else 'black')
        
        if int(key[4:7]) in core:
            CCcore.append(cc)
        else:
            CCouter.append(cc)
        
fig.tight_layout()
fig.savefig(os.path.join(directory,'met_vs_chi12_ILE_LEU.pdf'))
        

#%% Hop probability vs. state : VAL
cmap0=plt.get_cmap('tab20c')
cmap=[cmap0(0),cmap0(4),cmap0(8)]

fig,ax=plt.subplots(4,4)
fig.set_size_inches([9.5,10.5])
ax=ax.flatten()

counter=0
for key,value in index.items():
    if 'VAL' in key:
        ax0=ax[counter]
        counter+=1
        
        R=Rmet.T[value['MET']]
        fc1=frac_chi1.T[value['chi1']].T
        
        a=np.linalg.pinv(fc1.T)@R
        Rc=a@fc1
        
        cc=((R-R.mean())*(Rc-Rc.mean())).mean()/np.std(R)/np.std(Rc)
        
        yl=set_axis(R.max(),ax0)
        
        for k,color in enumerate(cmap):
            ax0.fill_between(t,fc1[:k].sum(0)*yl[1],fc1[:k+1].sum(0)*yl[1],color=color)
        ax0.plot(t,R,color='black')
        ax0.plot(t,Rc,color='white',linestyle='--',linewidth=2)
        
        if ax0.is_first_col():
            ax0.set_ylabel(r'$R_{hop}$ / ns$^{-1}$')
        # else:
        #     ax0.set_yticklabels('')
        if ax0.is_last_row():
            ax0.set_xlabel(r'$t$ / $\mu$s')
        else:
            ax0.set_xticklabels('')

        ax0.set_title(key[4:7]+key[:3])
        ax0.set_xlim([0,t[-1]])
        
        ax0.text(.5,yl[1]*0.8,f'cc = {cc:.2f}',color='white' if int(key[4:7]) in core else 'black')
        
        if int(key[4:7]) in core:
            CCcore.append(cc)
        else:
            CCouter.append(cc)
fig.tight_layout()
fig.savefig(os.path.join(directory,'met_vs_chi12_VAL.pdf'))
        

#%% chi2 hop probability vs. state : ILE and LEU

core=[231,239,241,264,267,275,277]
CCcore=list()
CCouter=list()

cmap0=plt.get_cmap('tab20c')
cmap=[cmap0(0),cmap0(1),cmap0(2),cmap0(4),cmap0(5),cmap0(6),cmap0(8),cmap0(9),cmap0(10)]

fig,ax=plt.subplots(3,3)
fig.set_size_inches([9.5,10.5])
ax=ax.flatten()

counter=0
for key,value in index.items():
    if 'chi2' in value and '_A' in key:
        ax0=ax[counter]
        counter+=1
        
        R=Rchi2.T[value['chi2']]
        fc12=frac_chi12.T[value['chi2']].T
        
        a=np.linalg.pinv(fc12.T)@R
        Rc=a@fc12
        
        cc=((R-R.mean())*(Rc-Rc.mean())).mean()/np.std(R)/np.std(Rc)
        
        yl=set_axis(R.max(),ax0)
        
        for k,color in enumerate(cmap):
            ax0.fill_between(t,fc12[:k].sum(0)*yl[1],fc12[:k+1].sum(0)*yl[1],color=color)
        ax0.plot(t,R,color='black')
        ax0.plot(t,Rc,color='white',linestyle='--',linewidth=2)
        
        if ax0.is_first_col():
            ax0.set_ylabel(r'$R_{hop}$ / ns$^{-1}$')
        # else:
        #     ax0.set_yticklabels('')
        if ax0.is_last_row():
            ax0.set_xlabel(r'$t$ / $\mu$s')
        else:
            ax0.set_xticklabels('')
            
        ax0.set_title(key[4:7]+key[:3])
        ax0.set_xlim([0,t[-1]])
        
        ax0.text(.5,yl[1]*.8,f'cc = {cc:.2f}',color='white' if int(key[4:7]) in core else 'black')
        
        if int(key[4:7]) in core:
            CCcore.append(cc)
        else:
            CCouter.append(cc)
fig.tight_layout()            
fig.savefig(os.path.join(directory,'chi2_vs_chi12_ILE_LEU.pdf'))
            
#%% chi1 hop probability vs. state : ILE / LEU

core=[231,239,241,264,267,275,277]
CCcore=list()
CCouter=list()

cmap0=plt.get_cmap('tab20c')
cmap=[cmap0(0),cmap0(1),cmap0(2),cmap0(4),cmap0(5),cmap0(6),cmap0(8),cmap0(9),cmap0(10)]

fig,ax=plt.subplots(3,3)
fig.set_size_inches([9.5,10.5])
ax=ax.flatten()

counter=0
for key,value in index.items():
    if 'chi2' in value and '_A' in key:
        ax0=ax[counter]
        counter+=1
        
        R=Rchi1.T[value['chi1']]
        fc12=frac_chi12.T[value['chi2']].T
        
        a=np.linalg.pinv(fc12.T)@R
        Rc=a@fc12
        
        cc=((R-R.mean())*(Rc-Rc.mean())).mean()/np.std(R)/np.std(Rc)
        
        yl=set_axis(R.max(),ax0)
        
        for k,color in enumerate(cmap):
            ax0.fill_between(t,fc12[:k].sum(0)*yl[1],fc12[:k+1].sum(0)*yl[1],color=color)
        ax0.plot(t,R,color='black')
        ax0.plot(t,Rc,color='white',linestyle='--',linewidth=2)
        
        if ax0.is_first_col():
            ax0.set_ylabel(r'$R_{hop}$ / ns$^{-1}$')
        # else:
        #     ax0.set_yticklabels('')
        if ax0.is_last_row():
            ax0.set_xlabel(r'$t$ / $\mu$s')
        else:
            ax0.set_xticklabels('')
            
        ax0.set_title(key[4:7]+key[:3])
        ax0.set_xlim([0,t[-1]])
        
        ax0.text(.5,yl[1]*.8,f'cc = {cc:.2f}',color='white' if int(key[4:7]) in core else 'black')
        
        if int(key[4:7]) in core:
            CCcore.append(cc)
        else:
            CCouter.append(cc)
fig.tight_layout()
fig.savefig(os.path.join(directory,'chi1_vs_chi12_ILE_LEU.pdf'))

#%% chi1 probability vs. state : VAL
cmap0=plt.get_cmap('tab20c')
cmap=[cmap0(0),cmap0(4),cmap0(8)]

fig,ax=plt.subplots(2,4)
fig.set_size_inches([9.5,5.5])
ax=ax.flatten()

counter=0
for key,value in index.items():
    if 'VAL' in key and '_A' in key:
        ax0=ax[counter]
        counter+=1
        
        R=Rchi1.T[value['chi1']]
        fc1=frac_chi1.T[value['chi1']].T
        
        a=np.linalg.pinv(fc1.T)@R
        Rc=a@fc1
        
        cc=((R-R.mean())*(Rc-Rc.mean())).mean()/np.std(R)/np.std(Rc)
        
        yl=set_axis(R.max(),ax0)
        
        for k,color in enumerate(cmap):
            ax0.fill_between(t,fc1[:k].sum(0)*yl[1],fc1[:k+1].sum(0)*yl[1],color=color)
        ax0.plot(t,R,color='black')
        ax0.plot(t,Rc,color='white',linestyle='--',linewidth=2)
        
        if ax0.is_first_col():
            ax0.set_ylabel(r'$R_{hop}$ / ns$^{-1}$')
        # else:
        #     ax0.set_yticklabels('')
        if ax0.is_last_row():
            ax0.set_xlabel(r'$t$ / $\mu$s')
        else:
            ax0.set_xticklabels('')

        ax0.set_title(key[4:7]+key[:3])
        ax0.set_xlim([0,t[-1]])
        
        ax0.text(.5,yl[1]*.8,f'cc = {cc:.2f}',color='white' if int(key[4:7]) in core else 'black')
        
        if int(key[4:7]) in core:
            CCcore.append(cc)
        else:
            CCouter.append(cc)
fig.tight_layout()
fig.savefig(os.path.join(directory,'chi1_vs_chi1_VAL.pdf'))        
            
#%% comparison of hop rates
fig,ax=plt.subplots(4,4)
fig.set_size_inches([9.5,10.5])
ax=ax.flatten()
counter=0

i_core=list()

for key,value in index.items():
    if int(key[4:7]) in core:
        ax[counter].plot(t,Rmet.T[value['MET']])
        ax[counter].set_title(key[4:7]+key[:3])
        i_core.append(value['MET']/3)
        counter+=1
i_core=np.array(i_core,dtype=int)        
#%% Covariance matrix
R0=Rmet[:,::3].copy()
R0-=R0.mean(0)
R0/=np.std(R0,axis=0)

covar=(R0.T@R0)/R0.shape[0]

cc=covar/np.std(R0,axis=0)
cc=(cc.T/np.std(R0,axis=0)).T

cc-=np.eye(cc.shape[0])



fig,ax=plt.subplots(1,2)
ax[0].imshow(cc)
ax[1].imshow(cc[i_core][:,i_core])
