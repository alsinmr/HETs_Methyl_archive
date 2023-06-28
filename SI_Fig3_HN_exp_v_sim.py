#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:24:11 2023

@author: albertsmith
"""



import pyDR

projHN=pyDR.Project('Projects/directHN')  #Backbone data project (exper+sim)

projHN['NMR']['proc'].plot(color='black',rho_index=range(3),errorbars=True)
(projHN['opt_fit']['HETs_4pw']+projHN['opt_fit']['HETs_MET_4pw']).plot(style='bar')

for a in projHN.plot_obj.ax:
    a.set_ylim([0,a.get_ylim()[1]])