#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 11:18:13 2022

@author: albertsmith
"""

"""
This is the main script for the paper:
    
    Title
    
    Authors
    
From here we run all processing and figure creation. Note that all data has been
pre-processed. The processing scripts are nonetheless provided, but the lines
corresponding to their running has been commented out.

We use runpy to run other scripts
"""

from runpy import run_path

#%% Process NMR data and process 6 MD simulations for comparison of dynamics
"""
Processes both backbone H–N dynamics and H–C methyl dynamics. 
H–N dynamics stored in project: "directHN"
H–C dynamics stored in project: "directHC"
"""
# _=run_path(path_name='direct_setup.py')   #Already run with results stored in archive

#%% Create Figure 1: Comparison of experiment and simulations
_=run_path(path_name='Fig1_compare_sims.py')

#%% Create Figure 2: Create chimera plots of total and frame separated motions
"""
First, we run the frame analysis. This has already been perfored and stored in
the project "frames"
"""
# _=run_path(path_name='frames_setup.py') #Already run with results stored in archive

"""
For this to work, one must specify the location of ChimeraX executable
Uncomment the next two lines and adjust the path accordingly
(this only needs to be run once, and will then save the location permanently)
"""
# from pyDR.chimeraX.chimeraX_funs import set_chimera_path
# set_chimera_path('/Applications/ChimeraX-1.2.5.app/Contents/MacOS/ChimeraX')

_=run_path(path_name='Fig2_chimera_plots.py')

#%% Create Figure 3: Plots comparing individual motions between experiment and MD sims
"""
First, we establish a linear relationship between 1/sigma^2 and log10(tau_hop) 
where sigma is the standard deviation of the rotation angle (degrees) for methyl 
libration and tau_hop is the correlation time of methyl hopping

This also creates plots of methyl hopping from experiment and simulation. However,
what is shown in the paper is a global fit of the experimental data
"""

_=run_path(path_name='Fig3A_met_hop_vs_lib.py',run_name='__main__')

"""
Next, we compare populations of chi1/chi2 rotamers extracted from experiment
and simulation
"""

_=run_path(path_name='Fig3C_plot_chi_pop.py',run_name='__main__')