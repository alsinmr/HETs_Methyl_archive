#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:49:08 2023

@author: albertsmith
"""

import pyDR
import os
from pyDR.PCA import PCA
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt

mddir='/Volumes/My Book/HETs/'

topo='HETs_SegB.pdb'
traj='MDSimulation/HETs_MET_4pw_SegB.xtc'

#%% Run PCA, save results for reload


if not(os.path.exists('PCA_results')):  #Either setup and run PCA, or re-load old run
    os.mkdir('PCA_results')
    
    pca=PCA(pyDR.MolSelect(topo=os.path.join(mddir,topo),
                           traj_files=os.path.join(mddir,traj),step=100)).\
                            select_atoms('name N C CA and (resid 225-245 or resid 261-281)')
    
    pca.runPCA()
    with open(os.path.join('PCA_results','covar.data'),'wb') as f:
        np.save(f,pca.CoVar,allow_pickle=False)
    with open(os.path.join('PCA_results','Lambda.data'),'wb') as f:
        np.save(f,pca.Lambda,allow_pickle=False)
    with open(os.path.join('PCA_results','PC.data'),'wb') as f:
        np.save(f,pca.PC,allow_pickle=False)
    with open(os.path.join('PCA_results','pcamp.data'),'wb') as f:
        np.save(f,pca.PCamp,allow_pickle=False)
    with open(os.path.join('PCA_results','mean.data'),'wb') as f:
        np.save(f,pca.mean,allow_pickle=False)
    copyfile(os.path.join(mddir,topo),os.path.join('PCA_results',topo))
    
else:
    pca=PCA(pyDR.MolSelect(topo=os.path.join(mddir,topo),
                           step=100)).select_atoms('name N C CA and (resid 225-245 or resid 261-281)')
    with open(os.path.join('PCA_results','covar.data'),'rb') as f:
        pca._covar=np.load(f,allow_pickle=False)
    with open(os.path.join('PCA_results','pcamp.data'),'rb') as f:
        pca._pcamp=np.load(f,allow_pickle=False)
    with open(os.path.join('PCA_results','mean.data'),'rb') as f:
        pca._mean=np.load(f,allow_pickle=False)
    with open(os.path.join('PCA_results','Lambda.data'),'rb') as f:
        pca._lambda=np.load(f,allow_pickle=False) 
    with open(os.path.join('PCA_results','PC.data'),'rb') as f:
        pca._PC=np.load(f,allow_pickle=False)
    pca._t=np.arange(0,10e3+.5,.5)



id0=np.load('DihedralData/indexfile.npy',allow_pickle=True).item()

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
met=np.mod(np.load('DihedralData/dih_MET_4pw_10μs.npy'),360)
chi1=np.mod(np.load('DihedralData/dih_chi1_MET_4pw_10μs.npy'),360)
chi2=np.mod(np.load('DihedralData/dih_chi2_MET_4pw_10μs.npy'),360)


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


#%% Determine hop times and binned rates
hop_met=np.diff(state_met,axis=0).astype(bool)
hop_chi1=np.diff(state_chi1,axis=0).astype(bool)
hop_chi2=np.diff(state_chi2,axis=0).astype(bool)

n=100
N=hop_met.shape[0]//n
dt=.005*n

Rmet=np.zeros([N,hop_met.shape[1]])    #Rates of hopping
Rchi1=np.zeros([N,hop_chi1.shape[1]])
Rchi2=np.zeros([N,hop_chi2.shape[1]])

for k in range(n):
    Rmet+=hop_met[k::n]/dt
    Rchi1+=hop_chi1[k::n]/dt
    Rchi2+=hop_chi2[k::n]/dt


#%% Correlate rates to PCA modes
fig,ax=plt.subplots(10,3)

core=[228,231,233,237,239,241,264,267,273,275,277]

c0=[plt.get_cmap('tab10')(0),plt.get_cmap('tab10')(1)]

for a,PCamp in zip(ax.T[0],pca.PCamp):
    cc=list()
    color=list()
    resi=list()
    for key,value in index.items():
        if 'MET' in value:
            resi.append(int(key[4:7]))
            color.append(c0[0] if resi[-1] in core else c0[1])
            cc.append(np.corrcoef(Rmet.T[value['MET']],PCamp[::PCamp.shape[0]//N])[0][1])
    a.bar(range(len(resi)),cc,color=color)
    a.set_xticks(range(len(resi)))
    a.set_xticklabels(resi,rotation=90)

for a,PCamp in zip(ax.T[1],pca.PCamp):
    cc=list()
    color=list()
    resi=list()
    for key,value in index.items():
        if 'chi1' in value and key[-1]=='A':
            resi.append(int(key[4:7]))
            color.append(c0[0] if resi[-1] in core else c0[1])
            cc.append(np.corrcoef(Rchi1.T[value['chi1']],PCamp[::PCamp.shape[0]//N])[0][1])
    a.bar(range(len(resi)),cc,color=color)
    a.set_xticks(range(len(resi)))
    a.set_xticklabels(resi,rotation=90)
    

for a,PCamp in zip(ax.T[2],pca.PCamp):
    cc=list()
    color=list()
    resi=list()
    for key,value in index.items():
        if 'chi2' in value and key[-1]=='A':
            resi.append(int(key[4:7]))
            color.append(c0[0] if resi[-1] in core else c0[1])
            cc.append(np.corrcoef(Rchi2.T[value['chi2']],PCamp[::PCamp.shape[0]//N])[0][1])
    a.bar(range(len(resi)),cc,color=color)
    a.set_xticks(range(len(resi)))
    a.set_xticklabels(resi,rotation=90)




fig,ax=plt.subplots(5,2)
ax=ax.flatten()

core=[228,231,233,237,239,241,264,267,273,275,277]

c0=[plt.get_cmap('tab10')(0),plt.get_cmap('tab10')(1)]

for k,(a,PCamp) in enumerate(zip(ax.T,pca.PCamp)):
    cc=list()
    color=list()
    resi=list()
    label=list()
    for key,value in index.items():
        if 'MET' in value:
            resi.append(int(key[4:7]))
            label.append('' if len(resi)>1 and resi[-1]==resi[-2] else str(resi[-1]))
            color.append(c0[0] if resi[-1] in core else c0[1])
            cc.append(np.corrcoef(Rmet.T[value['MET']],PCamp[::PCamp.shape[0]//N])[0][1])
    a.bar(range(len(resi)),-np.array(cc),color=color)
    a.set_xticks(range(len(resi)))
    a.set_xticklabels(label,rotation=90)
    a.text(0,a.get_ylim()[0]+.1,f'PC{k}')
    a.plot(a.get_xlim(),[0,0],color='black')
    
    if k==1:
        ax1=plt.figure().add_subplot(111)
        ax1.bar(range(len(resi)),cc,color=color)
        ax1.set_xticks(range(len(resi)))
        ax1.set_xticklabels(label,rotation=90)
        ax1.text(0,a.get_ylim()[0]+.1,f'PC{k}')
 



#%% Check correlation times
nd=7
data_pca=pca.PCA2data()
data_pca.detect.r_auto(nd)
fit_pca=data_pca.fit().opt2dist(rhoz_cleanup=True)

FT=np.fft.fft(Rmet.T[::3],n=Rmet.shape[0]*2,axis=-1)
ct=np.fft.ifft(FT.conj()*FT,axis=-1).real.T[:Rmet.shape[0]].T
ct/=np.arange(Rmet.shape[0],0,-1)
ct=ct.T
ct-=Rmet.mean(0)[::3]**2
ct/=ct[0]
ct=ct.T

sens=pyDR.Sens.MD(t=dt*np.arange(ct.shape[1]))
data=pyDR.Data(R=ct,sens=sens)
data.Rstd=np.repeat([data.sens.info['stdev']],data.R.shape[0],axis=0).astype(float)

data.detect.r_auto(nd)
data.label=np.array(label)
fit=data.fit().opt2dist(rhoz_cleanup=True)

# Plot PC1 detector analysis vs average detector response for inward facing residues
ind=np.array([r in core for r in resi],dtype=bool)
rho_avg=fit.R[ind].mean(0)

ax=plt.figure().add_subplot(111)
ax.bar(range(nd),fit_pca.R[1],color=[plt.get_cmap('tab10')(k) for k in range(nd)])
ax.plot(range(nd),rho_avg,color='black')


nice=pyDR.misc.disp_tools.NiceStr('(~{:q2})',unit='s')
lbls=[r'$\rho_'+f'{k}$\n '+nice.format(10**z0) for k,z0 in enumerate(fit.sens.info['z0'])]

ax.set_xticks(range(nd))
ax.set_xticklabels(lbls)
ax.figure.tight_layout()

# Plot detector analysis of all 10 PCs
fig,ax=plt.subplots(5,2)
ax=ax.flatten()
for k,a in enumerate(ax):
    for n in range(nd-1):
        a.bar(range(nd-1),fit_pca.R[k][:-1],color=[plt.get_cmap('tab10')(k) for k in range(nd)])
        a.set_title(f'PC {k}')
fig.set_size_inches([7,11])
fig.tight_layout()


#%% Save pictures of principal componets

for k in range(10):
    pca.project.chimera.close()
    pca.chimera(n=k,std=3)
    pca.project.chimera.command_line(['turn z 25','turn y -90','turn z 10','set bgColor white'])
    # pca.project.chimera.savefig(f'/Users/albertsmith/Documents/GitHub/HETs_methyl_archive/PCA_results/Figures/pc{k}.png',
    #                             options='transparentBackground True')


