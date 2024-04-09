# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:50:28 2023

@author: Hayley B. Caldwell 

DNap Relationship (Sigma) 02: IAF 
"""

import mne
import os
import os.path as op
import glob
from philistine.mne import savgol_iaf

# set working directory to where the raw EEG files are located 
os.chdir('E:\\DNap\\EEG') 
iaf_files = glob.glob('*_rs1_*.vhdr') 

#Electrodes to consider for IAF calculation
# = P1, Pz, P2, PO3, POz, PO4, O2, Oz, O2
electrodes = ['P3','P4','O1','O2','P7','P8','Pz']
#picks = [46,47,48,55,56,57,61,62,63]
#Frequency bands to adjust
bands = ["alpha","l_alpha","u_alpha","theta", "sigma","beta","alphabeta"]

def get_freq_band_limits(band, paf):
    """Adjust frequency bands using IAF.

    Uses the golden mean-based algorithm outlined in Klimesch (2012).
    """
    #Golden mean constant
    g = 1.618
    paf_delta = paf/4
    paf_theta = paf/2
    paf_beta = paf*2
    paf_gamma = paf*4
    if band == "alpha":
        lower = round(paf_theta*g,1)
        upper = round(paf_beta/g,1)
    elif band == "l_alpha":
        lower = round(paf_theta*g,1)
        upper = paf
    elif band == "u_alpha":
        lower = paf
        upper = round(paf_beta/g,1)
    elif band == "theta":
        lower = round(paf_delta*g,1)
        upper = round(paf/g,1)
    elif band == "sigma":
        lower = round(paf_beta/g,1) ### unsure 
        upper = round(paf*g,1) ### unsure
    elif band == "beta":
        lower = round(paf*g,1)
        upper = round(paf_gamma/g,1)
    elif band == "alphabeta":
        lower = round(paf_theta*g,1)
        upper = round(paf_gamma/g,1)
    
    return lower, upper

outfile = open('sigma\\processed\\iaf_long.txt','w')
header = "subj"+"\t"+"cond"+"\t"+"measure"+"\t"+"value"+"\n"
outfile.write(header)

for i in iaf_files:
    print("processing file "+i)
    subj = '_'.join(i.split('_')[:1])
    cond = '_'.join(i.split('_')[3:])
    cond = '_'.join(cond.split('.')[:1])
    
    raw = mne.io.read_raw_brainvision(i, preload=True)
    
    if subj == "21" and cond == "ret":
        raw.crop(tmin=0, tmax=120,
                             include_tmax=True, verbose=None)
    else: 
        pass

    #Standardise electrode names and select picks for IAF calculation
    #ch_names = dict()
    #for i,c in enumerate(raw.ch_names):
     #   ch_names[raw.ch_names[i]] = standardize_ch_name(c)
    #raw.rename_channels(ch_names)

    picks = list()
    for e in electrodes:
        index = raw.ch_names.index(e)
        picks.append(index)

    #Get IAF
    paf, cog, ablimits = savgol_iaf(raw, picks=picks, fmin=7, fmax=13)

    outfile.write(subj+"\t"+cond+"\t"+"paf\t"+str(paf)+"\n")
    outfile.write(subj+"\t"+cond+"\t"+"cog\t"+str(cog)+"\n")

    #Calculate adjusted frequency band limits
    for b in bands:
        band_lower = b + "_" + "lower"      
        band_upper = b + "_" + "upper"      
        try:
            lower,upper = get_freq_band_limits(b,paf)
        except (TypeError):
            lower,upper = get_freq_band_limits(b,10)
            
        outfile.write(subj+"\t"+cond+"\t"+band_lower+"\t"+str(lower)+"\n")
        outfile.write(subj+"\t"+cond+"\t"+band_upper+"\t"+str(upper)+"\n")

outfile.close()