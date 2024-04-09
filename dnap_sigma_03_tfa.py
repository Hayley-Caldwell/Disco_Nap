# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:26:25 2023

@author: Hayley B. Caldwell 

DNap Relationship (Sigma) 03: Time-Frequency Analysis  
"""

import sys
import mne
import numpy as np
import os.path as op
import glob
import pandas as pd
import csv
import os
from path import Path

from mne.time_frequency import tfr_morlet

def get_channel_name(epochs,ch_number):
    ch_name = epochs.ch_names[ch_number]
    return ch_name

def drop_position(event_id):
    new_id = event_id[0:6]+event_id[8:]
    return new_id

def compute_power(tf_all,epochs,freqs,ncycles,s_no,c_no):
    tf = tfr_morlet(epochs[c_no], freqs=freqs,n_cycles=ncycles, return_itc=False, average=False)
    av = tf.data
    arr = np.column_stack(list(map(np.ravel, np.meshgrid(*map(np.arange, av.shape), indexing="ij"))) + [av.ravel()])
    df = pd.DataFrame(arr, columns = ['epoch','channel', 'frequency', 'time','power'])
    #add subject and condition info
    df['subj'] = s_no
    df['cond'] = c_no
    tf_all = tf_all.append(df)
    return tf_all

def add_windows(tf,epoch_tmin,epoch_tmax,windows,sfreq):
    zero_ms = (0-epoch_tmin)*sfreq

    tf['win'] = 'none'
    for w in windows:
        name = w
        start = windows[w][0]
        stop = windows[w][1]
        if start < 0:
            start_sample = int((zero_ms - abs(start))/1000*sfreq)
        else:
            start_sample = int((zero_ms + start)/1000*sfreq)
        if stop < 0:
            stop_sample = int((zero_ms - abs(stop))/1000*sfreq)
        else:
            stop_sample = int((zero_ms + stop)/1000*sfreq)

        mask = (tf['time']>(start_sample-1)) & (tf['time']<(stop_sample-1))
        tf.loc[mask,'win'] = name

    mask_drop = tf['win'] != 'none'
    tf_wins = tf.loc[mask_drop,]
    return tf_wins

# set working directory to where the pre-processed EEG files are 
os.chdir('E:\\DNap\\EEG\\sigma\\processed')

#participants to exclude
exclude = []

#get lists of processed input files
epoch_files = glob.glob('*_epo.fif.gz') 
epoch_files_copy = glob.glob('*_epo.fif - Copy.gz')

#toggle this to true if you want to overwrite already processed files
#toggle this to false if you want to skip already processed files
compute_from_scratch = False 

tf_list = []

#Define frequency bands and windows of interest
bands = ["theta", "sigma"]
# matching the window to the 30 second window 
windows = {"Prestim":(-200,-0.5),
        "Event":(0, 30000)
        }

#Read in IAF data for individual frequency definition
iaf_info = pd.read_table('iaf_long.txt',dtype = {'subj':str, 'measure':str, 'value':np.float64}, na_values = 'None')
#Calculate mean across pre- and post-exp sessions
iaf_info_means = iaf_info.groupby(['subj','cond','measure'],as_index=False).agg('mean')

# extract theta and sigma power per epoch, per participant
for e in epoch_files:
    s_no = e.split('_')[0]
    a = e.split('_')[1]
    if e not in exclude:
        print("processing subject no." + s_no + ". Condition:" + a)
        
        if op.exists('power\\' + s_no + "_" + a + '_sigma_sigma.csv'):
            if not(compute_from_scratch):
                print('skipping participant' + s_no + "-condition: " + a +': file already processed')
                continue  
        
        #read in epochs 
        epochs = mne.read_epochs(e, preload=True)
        
        #initialise dataframe for tf output
        tf_all = pd.DataFrame()
        
        # calculate power per band 
        for b in bands:
            print("processing "+ b + " band")
            #Define frequency band limits based on IAF 
            band_lower = b + "_" + "lower"      
            band_upper = b + "_" + "upper" 
            lower = float(iaf_info_means.value[(iaf_info_means['subj'] == s_no) & (iaf_info_means['measure'] == band_lower) & (iaf_info_means['cond'] == a)])
            upper = float(iaf_info_means.value[(iaf_info_means['subj'] == s_no) & (iaf_info_means['measure'] == band_upper) & (iaf_info_means['cond'] == a)])
            
            freqs = np.linspace(lower,upper,5)
            ncycles = freqs/4

            tf_list = []
            
            # calculate power and output to a dataframe
            print("processing")
            tf = mne.time_frequency.tfr_array_morlet(epochs.get_data(), sfreq=epochs.info['sfreq'], freqs=freqs,n_cycles=ncycles, output='power')
            df = pd.DataFrame(np.column_stack(list(map(np.ravel, np.meshgrid(*map(np.arange, tf.shape), indexing="ij"))) + [tf.ravel()]), columns = ['epoch','channel', 'frequency', 'time','power'])
            df['subj'] = s_no
            df['band'] = b
            df['condition'] = a
            tf_list.append(df)
                
            # export the current dataframe 
            tf_all = pd.concat(tf_list)
            tf_all_wins = add_windows(tf_all,epochs.tmin,epochs.tmax,windows,epochs.info['sfreq'])
            tf_all_sel = tf_all_wins.drop(['frequency','time'],axis=1)
            tf_means = tf_all_sel.groupby(['epoch','channel','subj', 'condition', 'win','band'],as_index=False).agg('mean')
            tf_means['ch_name'] = tf_means.apply(lambda row: get_channel_name(epochs,int(row['channel'])), axis=1)
            filepath = 'power\\'+s_no+'_'+a+'_'+b+'_sigma.csv'
            tf_means.to_csv(filepath,index=False)
            
            # add the current dataframe to a larger dataframe with all bands and participants 
            df_export = pd.DataFrame(tf_means)
            if op.isfile('sigma_power.csv'):
                df_export.to_csv('sigma_power.csv', sep = ',', mode = 'a', header = False,
                                 index = False)
            else:
                df_export.to_csv('sigma_power.csv', sep = ',', mode = 'a', header = True,
                                 index = False)
    