# -*- coding: utf-8 -*-

"""
Pre-processing, covariance-based artifact rejection, detection and plotting of
spindle and slow wave events, and cross-frequency coupling analyses.

Functions from YASA (Vallet, 2020; https://github.com/raphaelvallat/yasa)

Authors: Alex Chatburn (the Serpent King)

"""
import mne
import yasa
import os
import os.path as op
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
import pingouin as pg
from tensorpac import Pac
from pandas import read_csv

sns.set(style='white', font_scale=1.2)

# set the working directoy where the raw eeg files are located
os.chdir('D:\\DNap\\EEG')

# create list of sleep EEG files
# need to create two lists for .vhdr files and files that were appended
sleep_files = glob.glob('*_int_nap.vhdr')

#toggle this to true if you want to overwrite already processed files
#toggle this to false if you want to skip already processed files
compute_from_scratch = False

## ---------------------------------------------------------------------------
## Basic Pre-Processing
## ---------------------------------------------------------------------------

# loop through each file for each subject
# be sure to change 'appended_files' to 'sleep_files' and vice versa
for s in sleep_files:
    print("processing file " + s)
    subj = op.split(s)[1][0:2] 
    
    if op.exists('processed/' + subj + '_nap' + '_raw.fif.gz'):
        if not(compute_from_scratch):
            print('skipping participant' + subj + ': file already exists')
            continue
        
    outfile = 'processed/' + subj + '_nap' + '_raw.fif.gz'
    
    # read in raw sleep EEG data
    raw = mne.io.read_raw_brainvision(s, eog=('E1','E2'), misc=('EMG1','EMG2','EMG3', 'ECG'), preload=True)
    
    # downsample to 100 Hz
    raw = raw.resample(100)
    
    # re-reference to linked mastoids
    raw = mne.io.set_eeg_reference(raw,['M1','M2'])[0]

    # apply basic pre-processing parameters
    raw = raw.filter(0.3, 30.,
                    l_trans_bandwidth='auto',
                     h_trans_bandwidth='auto',
                     filter_length='auto',
                     method='fir',
                     fir_window='hamming',
                     phase='zero',
                     n_jobs=2)
    
    # label mastoids and horizontal EOG as miscellaneous
    raw.set_channel_types({'M1':'misc','M2':'misc'})
    
    # save pre-processed EEG file
    raw.save(outfile,fmt='single',overwrite=True)
    
## ---------------------------------------------------------------------------
## Covariance-Based Artifact Rejection and Spectrogram Generation 
## ---------------------------------------------------------------------------
    
# list raw EEG and hypnogram files in chronological order so they are matched
# doesn't work. nothing works. need help.
os.chdir('D:\\DNap\\EEG\\processed\\')

art_files = glob.glob(os.path.join('*_nap_raw.fif.gz'))

# set parameters for analysis
sf_art   = 1/5
sf_hypno = 1/30
sf       = 100

# create a list of the channels we want to include
chans = ['Fz','F3','F4','Cz','C3','C4','Pz','P3','P4','O1','O2']

# toggle this to true if you want to overwrite the processed files
compute_from_scratch = False

# list the cases here that you want to process  
process_cases = ["30"]

# load data and hypnogram
for a in sorted(art_files, key=lambda s: s.lower()): 
    print("processing file " + a)
    subj = op.split(a)[1][0:2] 
    
    if subj not in process_cases:
        if not(compute_from_scratch):
            print('skipping participant ' + subj)
            continue
    
    f = mne.io.read_raw_fif(a)
    data = f.get_data(picks=['Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'O1', 'O2'])
    hypno_file = 'D:\\DNap\\EEG\\processed\\' + subj + '_hyp.csv'
    #hypno = yasa.load_profusion_hypno(hypno_file, replace=True)
    hypno = pd.read_csv(hypno_file)
    hypno = hypno.squeeze('columns') 
    
    # up sample hypnogram to match sampling rate of data (100 Hz)
    hypno = yasa.hypno_upsample_to_data(hypno, sf_hypno, data, sf)

    # run artifact rejection based on z scores
    art, zscores = yasa.art_detect(f, sf, window=5, hypno=hypno, 
                           include=(1, 2, 3, 4), method='covar', 
                           threshold=3, verbose='info')


    art_up = yasa.hypno_upsample_to_data(art, sf_art, f, sf)

    # Add -1 to hypnogram to indicate rejected epochs
    hypno_with_art = hypno.copy()
    hypno_with_art[art_up] = -1

    # plot the whole night of sleep and save the figure into folder 'spectrogram'
    plot_1 = yasa.plot_spectrogram(data[0, :], sf, hypno, cmap='viridis',
                           trimperc=5)
    sns.despine()
    plt.savefig('spectrogram/' + subj + '_hypno.png', dpi=300)

    # adapt scaling of data by converting to microvolts (uV)
    data = data * 1e6

## ---------------------------------------------------------------------------
## Spindle Detection
## ---------------------------------------------------------------------------

    # run spindle detection algorithm for stage 2 and sws (N2, N3)
    sp = yasa.spindles_detect(data, sf, ch_names=chans, hypno=hypno_with_art, 
                          include=(2, 3))

    # extract spindle metrics and add subject code to data structure
    sp_data = sp.summary(grp_chan=True, grp_stage=True, aggfunc='median').round(3)
    sp_data['subj'] = subj

    # save output file for each subject into folder 'spindle'
    sp_data.to_csv('spindle/' + subj + '_spindle.csv', header = True)

    # plot average spindle and save figure to folder 'spindle'
    ax = sp.plot_average(center='Peak', time_before=0.8, time_after=0.8, 
                     filt=(12, 16), ci=None, legend=False)
    sns.despine()
    plt.savefig('spindle/' + subj + '_spindle.png', dpi=300)
    
    # append each subject to grand .csv file
    df_export = pd.DataFrame(sp_data)
    if op.isfile('slow_waves.csv'):
        df_export.to_csv('spindles.csv', sep=',', mode='a', header=False)
    else:
        df_export.to_csv('spindles.csv', sep=',', mode='a', header=True)

## ---------------------------------------------------------------------------
## Slow Wave Detection
## ---------------------------------------------------------------------------

    # run slow wave detection algorithm for sws (N3)
    sw = yasa.sw_detect(data, sf, ch_names=chans, hypno=hypno_with_art, 
                    include=(3))

    # extract slow wave metrics and add subject code to data structure
    so_data = sw.summary(grp_chan=True, grp_stage=True, aggfunc='median').round(3)
    so_data['subj'] = subj
    print(sw.summary().shape[0], 'slow-waves detected.')

    # save output file for each subject into folder 'so'
    so_data.to_csv('so/' + subj + '_so.csv', header = True)

    # plot the slow waves with confidence intervals and save to folder 'so'
    ax2 = sw.plot_average(center="Start", time_before=2.5, time_after=2.5, 
                      legend = False)
    sns.despine()
    plt.savefig('so/' + subj + '_SW.png', dpi=300)

    # append each subject to grand .csv file
    df_export = pd.DataFrame(so_data)
    if op.isfile('slow_waves.csv'):
        df_export.to_csv('slow_waves.csv', sep=',', mode='a', header=False)
    else:
        df_export.to_csv('slow_waves.csv', sep=',', mode='a', header=True)
         
## ---------------------------------------------------------------------------
## Band Power Analysis
## ---------------------------------------------------------------------------
    
    power = yasa.bandpower(data, sf=sf, hypno=hypno_with_art, ch_names=chans, 
                            include=(2,3,4))
    power['subj'] = subj
    
    # append each subject to grand .csv file
    df_export = pd.DataFrame(power)
    if op.isfile('power.csv'):
         df_export.to_csv('power.csv', sep=',', mode='a', header=False)
    else:
         df_export.to_csv('power.csv', sep=',', mode='a', header=True)


## ---------------------------------------------------------------------------
## Slow Wave and Spindle Coupling
## ---------------------------------------------------------------------------

    # only use channel Cz to reduce computational complexity
    data_cz = data[3, :].astype(np.float64)
    print(data_cz.shape, np.round(data_cz[0:5], 3))

    # run slow wave and spindle detection function on stage 2 and sws (N2, N3)
    coup = yasa.sw_detect(data_cz, sf, hypno=hypno, include=(2, 3), 
                    coupling=True)#, freq_sp=(12, 16))

    # create data structure containing each coupling event
    events = coup.summary()

    # group data by sleep stage
    out = coup.summary(grp_stage=True).round(3)

    # add column for subject code
    out['subj'] = subj
    
    # append each subject to grand .csv file
    df_export = pd.DataFrame(out)
    if op.isfile('coupling.csv'):
        df_export.to_csv('coupling.csv', sep=',', mode='a', header=False)
    else:
        df_export.to_csv('coupling.csv', sep=',', mode='a', header=True)
        
    # plot circular histogram to visualise coupling
    plt.figure()
    circ2 = pg.plot_circmean(events['PhaseAtSigmaPeak'])
    sns.despine()
    plt.savefig('coupling/' + subj + '_circ.png', dpi=300)
    plt.close();

    print('Circular mean: %.3f rad' % pg.circ_mean(events['PhaseAtSigmaPeak']))
    print('Vector length: %.3f' % pg.circ_r(events['PhaseAtSigmaPeak']))

    # distribution of ndPAC (coupling strength) values: 
    plt.figure()
    hist = events['ndPAC'].hist()
    sns.despine()
    plt.savefig('coupling/' + subj + '_histogram.png', dpi=300)
    plt.close();

    # this should be close to the vector length that we calculated above
    events['ndPAC'].mean()
    
    # calculate data-driven phase amplitide coupling (PAC)

    # segment N3 sleep into 15-seconds non-overlapping epochs
    _, data_cz_N3 = yasa.sliding_window(data_cz[hypno == 3], sf, window=15)

    # we end up with x number of epochs of 15-seconds
    data_cz_N3.shape

    # first, let's define our array of frequencies for phase and amplitude
    f_pha = np.arange(0.125, 4.25, 0.25)  # frequency for phase
    f_amp = np.arange(7.5, 25.5, 0.5)     # frequency for amplitude

    f_pha, f_amp

    # now let's calculate the comodulogram 
    sns.set(font_scale=1.1, style='white')

    # define a PAC object
    p = Pac(idpac=(1, 0, 0), f_pha=f_pha, f_amp=f_amp, verbose='WARNING')

    # filter the data and extract the PAC values 
    xpac1 = p.filterfit(sf, data_cz_N3)

    # plot the comodulogram
    plt.figure()
    como1 = p.comodulogram(xpac1.mean(-1), title=str(p), vmin=0, plotas='imshow')
    sns.despine()
    plt.savefig('coupling/' + subj + '_coupling_mvl.png', dpi=300)
    plt.close();
    
    # extract PAC values into a data frame - no need to save this for now
    df_pac = pd.DataFrame(xpac1.mean(-1), columns=p.xvec, index=p.yvec)
    df_pac.columns.name = 'FreqPhase'
    df_pac.index.name = 'FreqAmplitude'
    df_pac.head(20).round(2)

    # change the idpac argument to calculate modulation index rather than mvl

    # define a PAC object
    p2 = Pac(idpac=(2, 0, 0), f_pha=f_pha, f_amp=f_amp, verbose='WARNING')

    # filter the data and extract the PAC values
    xpac2 = p2.filterfit(sf, data_cz_N3)

    # plot the comodulogram and save it
    plt.figure()
    como2 = p2.comodulogram(xpac2.mean(-1), title=str(p2), plotas='imshow')
    sns.despine()
    plt.savefig('coupling/' + subj + '_coupling_Tort.png', dpi=300)
    plt.close();

    # extract PAC values into a data frame - no need to save this for now
    df_pac2 = pd.DataFrame(xpac2.mean(-1), columns=p2.xvec, index=p2.yvec)
    df_pac2.columns.name = 'FreqPhase'
    df_pac2.index.name = 'FreqAmplitude'
    df_pac2.round(3)

## ---------------------------------------------------------------------------
## Grand Average Plots
## ---------------------------------------------------------------------------

# read in data file that contains all subjects
coup_cnt = read_csv("coupling.csv")

# plot circular histogram with all subjects
plt.figure()
circ2 = pg.plot_circmean(coup_cnt['PhaseAtSigmaPeak'],
                         kwargs_markers=dict(color='k',mfc='r'),
                         kwargs_arrow=dict(ec='r', fc='r'))
sns.despine()
plt.savefig('coupling/' + '_coupling_average_NREM.png', dpi=300)

# Get names of indexes for which column Stage has value 2
indexNames = coup_cnt[ coup_cnt['Stage'] == 3 ].index
# Delete these row indexes from dataFrame
coup_cnt.drop(indexNames , inplace=True)

# plot circular histogram with all subjects
plt.figure()
circ2 = pg.plot_circmean(coup_cnt['PhaseAtSigmaPeak'],
                         kwargs_markers=dict(color='k',mfc='r'),
                         kwargs_arrow=dict(ec='r', fc='r'))
sns.despine()
plt.savefig('coupling/' + '_coupling_average_N3.png', dpi=300)
