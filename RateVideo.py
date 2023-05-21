#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:47:29 2023

@author: albertsmith
"""

import numpy as np
import pyDR
import os


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


    
    
#%% Selection object for plotting the rate data
proj=pyDR.Project()  #Dummy project (for chimeraX usage)

mddir='/Volumes/My Book/HETs/'

# topo='HETs_SegB.pdb'
# traj='MDSimulation/HETs_MET_4pw_SegB.xtc'

topo='HETs_3chain.pdb'
traj='MDSimulation/HETs_MET_4pw.xtc'

#%% Methyl selection
molsys=pyDR.MolSys(topo=os.path.join(mddir,topo),traj_files=os.path.join(mddir,traj),step=n,project=proj)
sel_met=pyDR.MolSelect(molsys)
uni=sel_met.molsys.uni

sel1=[]
sel2=[]
repr_sel=[]
R=[]

for key,value in index.items():
    for k,v in value.items():
        if k=='MET':
            if key[:3]=='ILE' and key[-1]=='A':
                sel1.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CD')[0])
                sel2.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name HD1')[0])
                repr_sel.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CD HD1 HD2 HD3'))
                R.append(Rmet.T[v])
            if key[:3]=='ILE' and key[-1]=='B':
                sel1.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CG2')[0])
                sel2.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name HG21')[0])
                repr_sel.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CG2 HG21 HG22 HG23'))
                R.append(Rmet.T[v])
            if key[:3]=='LEU' and key[-1]=='A':
                sel1.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CD1')[0])
                sel2.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name HD11')[0])
                repr_sel.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CD1 HD11 HD12 HD13'))
                R.append(Rmet.T[v])
            if key[:3]=='LEU' and key[-1]=='B':
                sel1.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CD2')[0])
                sel2.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name HD21')[0])
                repr_sel.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CD2 HD21 HD22 HD23'))
                R.append(Rmet.T[v])
            if key[:3]=='VAL' and key[-1]=='A':
                sel1.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CG1')[0])
                sel2.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name HG11')[0])
                repr_sel.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CG1 HG11 HG12 HG13'))
                R.append(Rmet.T[v])
            if key[:3]=='VAL' and key[-1]=='B':
                sel1.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CG2')[0])
                sel2.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name HG21')[0])
                repr_sel.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CG2 HG21 HG22 HG23'))
                R.append(Rmet.T[v])
            if key[:3]=='ALA':
                sel1.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CB')[0])
                sel2.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name HB1')[0])
                repr_sel.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CB HB1 HB2 HB3'))
                R.append(Rmet.T[v])
xmet=np.array(R)
sel_met._sel1=np.array(sel1,dtype=object)
sel_met._sel2=np.array(sel2,dtype=object)
_=sel_met.repr_sel
for k,rs in enumerate(repr_sel):sel_met.repr_sel[k]=rs

#%% Chi2 selection
sel_chi2=pyDR.MolSelect(molsys)

sel1=[]
sel2=[]
repr_sel=[]
R=[]           
            
            
for key,value in index.items():
    for k,v in value.items():            
        if k=='chi2':
            if key[:3]=='ILE':
                sel1.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CG1')[0])
                sel2.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name HG11')[0])
                repr_sel.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CG1 HG11 HG12 CD'))
                R.append(Rmet.T[v])
            if key[:3]=='LEU' and key[-1]=='A':
                sel1.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CG')[0])
                sel2.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name HG')[0])
                repr_sel.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CG HG CD1 CD2'))
                R.append(Rmet.T[v])

xchi2=np.array(R)
sel_chi2._sel1=np.array(sel1,dtype=object)
sel_chi2._sel2=np.array(sel2,dtype=object)
_=sel_chi2.repr_sel  #Call this to pre-allocate it
for k,rs in enumerate(repr_sel):sel_chi2.repr_sel[k]=rs

#%% Chi1 selection
sel_chi1=pyDR.MolSelect(molsys)

sel1=[]
sel2=[]
repr_sel=[]
R=[]
            
for key,value in index.items():
    for k,v in value.items():
        if k=='chi1':
            if key[:3] in ['ILE','VAL'] and key[-1]=='A':
                sel1.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CB')[0])
                sel2.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name HB')[0])
                repr_sel.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CB HB CG1 CG2'))
                R.append(Rmet.T[v])
            if key[:3]=='LEU' and key[-1]=='A':
                sel1.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CB')[0])
                sel2.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name HB1')[0])
                repr_sel.append(uni.select_atoms(f'segid B and resid {key[4:7]} and name CB HB1 HB2 CG'))
                R.append(Rmet.T[v])

xchi1=np.array(R)
sel_chi1._sel1=np.array(sel1,dtype=object)
sel_chi1._sel2=np.array(sel2,dtype=object)
_=sel_chi1.repr_sel  #Call this to pre-allocate it
for k,rs in enumerate(repr_sel):sel_chi1.repr_sel[k]=rs


#%% Function to encode data onto HET-s molecule
def plot_data(t_index:int):
    # proj.chimera.close()
    for k in proj.chimera.CMX.valid_models(proj.chimera.CMXid):
        if k!=1:proj.chimera.command_line(f'close #{k}')
    molsys.make_pdb(t_index)
    
    sel_met.chimera()
    proj.chimera.command_line(
        ['~show','color #2 grey target a','color #2/A,C grey','color #1/B:ILE,LEU,VAL,ALA tan target a'])
    
    sel_met.chimera(x=xmet.T[t_index]/xmet.max(),color=[255,0,0,255])
    # sel_chi2.chimera(x=xchi2.T[t_index]/xchi2.max(),color=[0,255,0,255])
    # sel_chi1.chimera(x=xchi1.T[t_index]/xchi1.max(),color=[0,0.,255,255])
    proj.chimera.command_line(
        ['~show','show #3-5/B:ILE,LEU,VAL,ALA','show #2/A:261-281|#2/C:225-245',
         'color #2 grey target b',
         '~ribbon','setattr #1-5 residue is_strand false',
         'setattr :226-234,236-246,262-270,272-282 residue is_strand true',
         'ribbon #2/A:255-290|#2/C:218-255|#2/B',
         'transparency /A,C 30 target ra'])

proj.chimera.close()
sel_met.chimera()

plot_data(0)

proj.chimera.command_line(['lighting full','graphics silhouettes true','set bgColor white',
                           'turn z 25','turn y -90','turn z 10',
                           'move x 5'])

    
for t_index,_ in enumerate(molsys.traj[:900]):
    plot_data(t_index)
    proj.chimera.savefig(f'images/image{t_index:03d}.png',options='height 450 width 650')
    while not(os.path.exists(f'images/image{t_index:03d}.png')):pass
    
    
import cv2
size=None    
for t_index,_ in enumerate(molsys.traj[:900]):    
    img=cv2.imread(f'images/image{t_index:03d}.png')
    if size is None:
        size=(img.shape[1],img.shape[0])
        out=cv2.VideoWriter('video.mp4',cv2.VideoWriter_fourcc(*'mp4v'),15,size)
    out.write(img)
    print(f'{t_index} out of ')
out.release()    
    
