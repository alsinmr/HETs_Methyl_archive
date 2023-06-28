#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:31:07 2023

@author: albertsmith
"""

import numpy as np
import matplotlib.pyplot as plt


index=np.load('DihedralData/indexfile.npy',allow_pickle=True).item()
            

#%% Load data
chi1=np.mod(np.load('DihedralData/dih_chi1_MET_4pw_10μs.npy'),360)
chi2=np.mod(np.load('DihedralData/dih_chi2_MET_4pw_10μs.npy'),360)


#%% Determine state of each angle (0, 1, or 2)
ref0=np.mod(chi1,120).mean(1)
ref_chi1=np.concatenate(([ref0],[ref0+120],[ref0+240]),axis=0)
ref0=np.mod(chi2,120).mean(1)
ref_chi2=np.concatenate(([ref0],[ref0+120],[ref0+240]),axis=0)

state_chi1=np.argmin([np.abs(chi1.T-ref) for ref in ref_chi1],axis=0).T
state_chi2=np.argmin([np.abs(chi2.T-ref) for ref in ref_chi2],axis=0).T


 
#%% First, calculate the state index, determine total number of states
core=[228,231,233,237,239,241,264,267,273,275,277]
pocket=[]
core_i=[]
state=[]
resids=[]
mult=[]
for key,value in index.items():
    if key[:3]!='ALA':
        state.append(state_chi1[value['chi1']])
        mult.append(3)
        resids.append(int(key[-3:]))
        core_i.append(resids[-1] in core)
        if 'chi2' in value:
            state[-1]+=3*state_chi2[value['chi2']]
            mult[-1]*=3

state=np.array(state).T       
mult=np.array(mult)
core=np.array(core_i,dtype=bool)
not_core=np.logical_not(core)
resids=np.array(resids)


def calcS(count):
    R=8.314462
    if not(hasattr(count,'__len__')):
        return R*np.log(count)
    P=count/count.sum()
    return -R*(P*np.log(P)).sum()

#%% Calculate the maximum possible entropy

maxS=[calcS(mult[core].prod()),calcS(mult[not_core].prod()),calcS(mult.prod())]

#%% Calculate sum of entropy from individual residues
count_R=np.array([np.unique(s,return_counts=True)[1] for s in state.T])
S_R0=np.array([calcS(count) for count in count_R])
S_R=[S_R0[core].sum(),S_R0[not_core].sum(),S_R0.sum()]

#%% Calculate the total entropies
S=[]
for id0 in [core,not_core,np.ones(len(core),dtype=bool)]:
    q=np.cumprod([1,*mult[id0]])[:-1]
    state0=(state[:,id0]*q).sum(-1)
    count=np.unique(state0,return_counts=True)[1]
    S.append(calcS(count))


#%% Plot the results
ax=plt.subplots(1,1)[1]

w=.3
for k in range(3):
    ax.bar(w*(k-1)+np.arange(3),[maxS[k],S_R[k],S[k]],width=.33)
ax.set_xticks(range(3))
ax.set_xticklabels([r'max $\Delta S$',r'$\sum_\mathrm{i\in resids.}{\Delta S_i}$',r'Total $S$'])


#%% Pairwise S
cc=np.zeros([len(resids),len(resids)])
DelS=np.zeros([len(resids),len(resids)])
for p,(Sp,statep) in enumerate(zip(S_R0,state.T)):
    print(p)
    for q,(Sq,stateq) in enumerate(zip(S_R0,state.T)):
        if q<p:
            state0=statep+mult[p]*stateq
            count=np.unique(state0,return_counts=True)[1]
            S0=calcS(count)
            cc[p,q]=2*(Sp+Sq-S0)/(Sp+Sq)
            cc[q,p]=cc[p,q]
            DelS[p,q]=Sp+Sq-S0
            DelS[q,p]=DelS[p,q]
            
ax=plt.subplots(1,1)[1]
hdl=ax.imshow(cc,cmap='cool')
plt.colorbar(hdl)
ax.set_xticks(range(len(resids)))
ax.set_xticklabels(resids,rotation=90)
ax.set_yticks(range(len(resids)))
ax.set_yticklabels(resids)

#%% Residuewise DelS
DS_res=[]
cc_res=[]
for p,Sp in enumerate(S_R0):
    i=np.ones(len(resids),dtype=bool)
    i[p]=False
    q=np.cumprod([1,*mult[i]])[:-1]
    state0=(state[:,i]*q).sum(-1)
    count=np.unique(state0,return_counts=True)[1]
    S0=calcS(count)
    DS_res.append(S[-1]-S0)
    cc_res.append((Sp+S0-S[-1])/Sp)
    
ax=plt.subplots(1,1)[1]
ax.bar(range(len(resids)),cc_res)
ax.set_xticks(range(len(resids)))
ax.set_xticklabels(resids,rotation=90)

ax=plt.subplots(1,1)[1]
ax.bar(range(len(resids)),S_R0)
ax.bar(range(len(resids)),DS_res)
ax.set_xticks(range(len(resids)))
ax.set_xticklabels(resids,rotation=90)


#%% Time dependence of total S and residue-specific S
S_t=[]
S_R_t=[]
q=np.cumprod([1,*mult[id0]])[:-1]

N=np.linspace(1,state.shape[0],100).astype(int)

t=N*.005

for n in N:
    state0=(state[:n]*q).sum(-1)
    count=np.unique(state0,return_counts=True)[1]
    S_t.append(calcS(count))
    
    count_R=np.array([np.unique(s,return_counts=True)[1] for s in state[:n].T])
    S_R_t.append(np.array([calcS(count) for count in count_R]).sum())
    
ax=plt.subplots(1,1)[1]
ax.plot(t/1e3,S_t)
ax.plot(t/1e3,S_R_t)
ax.set_xlabel(r'$t$ / $\mu$s')
ax.set_ylabel(r'$\Delta S$ / J*mol$^{-1}$*$K^{-1}$')
ax.legend([r'Total $\Delta $',r'$\sum_{i\in \mathrm{resids}}{\Delta S_i}$'])

