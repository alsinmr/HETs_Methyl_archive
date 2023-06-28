#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 14:15:43 2022

@author: albertsmith
"""

"""
This creates the plots for Figure 2 and SI Figures 4-8
"""


import pyDR
import os

proj=pyDR.Project('Projects/proj_exp')
exp=pyDR.Project('Projects/directHC')['NMR']['proc'][0]
exp.src_data=None
proj.append_data(exp)
#Set up file storage
if not(os.path.exists('CMX_frame_plots')):
    os.mkdir('CMX_frame_plots')
filename=os.path.join(os.getcwd(),'CMX_frame_plots/{}_rho{}.png')


proj.chimera.saved_commands=['~show','ribbon /B','turn y -105','turn z 10',
                             'turn x -45','turn z 5','view','zoom 1.4',
                             'lighting soft','graphics silhouettes true'] #These commands always executed


sub=proj['HETs_MET_4pw_SegB']-proj['Product']-proj['OuterHop']+proj['NMR']

for d in sub:
    for k in range(4):
        proj.chimera.close()
        scaling=1.5*(9 if (k>1 and ('MetHop' in d.title or 'Direct' in d.title or 'NMR' in d.title)) else 1)
        d.chimera(rho_index=k,scaling=scaling)
        
        if 'NMR' in d.title:
            proj.chimera.command_line('show /B:'+','.join([lbl[:3] for lbl in d.label]))
            sels=['(:'+lbl[:3]+('@HD*' if lbl[3:] in ['Leu','Ile'] else ('@HG*' if lbl[3:]=='Val' else '@HB*'))+')' for lbl in d.label]
        else:
            proj.chimera.command_line('show /B:'+','.join([lbl.split('_')[0] for lbl in d.label])+'&~H') #just show residues where we have data
            sels=['(:'+lbl.split('_')[0]+'@'+lbl.split('_')[1].replace('C','H')+'*)' for lbl in d.label]
        proj.chimera.command_line('show /B&'+'|'.join(sels))       
        title=d.source.additional_info
        if title is None:title='exper'
        proj.chimera.savefig(filename.format(title,k),'transparentBackground True')

#%% Plot frames validation
"SI Figure 4"
proj['HETs_MET_4pw_SegB']['Direct'].plot(style='bar')
proj['HETs_MET_4pw_SegB']['Product'].plot()