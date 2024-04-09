# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:10:24 2023

@author: Hayley B. Caldwell 

DNap Relationship (Sigma) 01: Pre-Processing 
"""


import mne
import os
import os.path as op
import glob
import seaborn as sns
import csv
from philistine.mne import abs_threshold, retrieve
from autoreject import AutoReject, get_rejection_threshold
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs
import matplotlib.pyplot as plt

# plots created will appear in a different window - dont worry, it runs
%matplotlib qt

# set working directory to where the function is located 
os.chdir('E:\\DNap\\Scripts')
from utils import compute_ica_correction

# set working directory to where the raw EEG files are located 
os.chdir('E:\\DNap\\EEG')

sns.set(style='white', font_scale=1.2)

##############################################################################
#                           Setup the basics                                 #
##############################################################################

# construct a montage to assign to the data
montage = mne.channels.make_standard_montage('standard_1020')

raw_files = glob.glob("*dnap_int_re*.vhdr")

# toggle this to true if you want to overwrite already processed files
# toggle this to false if you want to skip already processed files
compute_from_scratch = False

# rejection thresholds for ICA and artifact rejection
reject_ica = dict(eeg=150e-6)
reject = dict(eeg=150e-6, eog=250e-6)
flat = dict(eeg=5e-6)
tmin, tmax = -0.2, 1.2

event_id_ret = {
    # extras
    'fixation': 1,  'start-phase': 2,  'stop-phase': 3,
    'r1-start': 230,  'r1-stop': 231,  'r2-start': 232,  'r2-stop': 233,
    'r3-start': 234,  'r3-stop': 235,  'r4-start': 236,  'r4-stop': 237,
    'r5-start': 238,  'r5-stop': 239,  'r6-start': 240,  'r6-stop': 241,
    # items
    'edge_1': 10,  'seat_1': 11,  'feat_1': 12,  'valve_1': 13,
    'heel_1': 14,  'sword_1': 15,  'sketch_1': 16,  'chore_1': 17,
    'slang_1': 18,  'prince_1': 19,  'thumb_1': 20,  'mutt_1': 21,
    'scalp_1': 22,  'ranch_1': 23,  'lime_1': 24,  'ramp_1': 25,
    'gig_1': 26,  'smirk_1': 27,  'mulch_1': 28,  'root_1': 29,
    'graft_1': 30,  'dude_1': 31,  'watch_1': 32,  'pump_1': 33,
    'loop_1': 34,  'lobe_1': 35,  'map_1': 36,  'mall_1': 37,
    'kit_1': 38,  'slope_1': 39,  'scotch_1': 40,  'knee_1': 41,
    'loom_1': 42,  'pig_1': 43,  'strap_1': 44,  'verse_1': 45,
    'chain_1': 46,  'plan_1': 47,  'guise_1': 48,  'curb_1': 49,
    'dark_1': 50,  'road_1': 51,  'crud_1': 52,  'sling_1': 53,
    'hound_1': 54,  'fog_1': 55,  'block_1': 56,  'fate_1': 57,
    'spleen_1': 58,  'pile_1': 59,  'fluff_1': 60,  'gull_1': 61,
    'jive_1': 62,  'tooth_1': 63,  'shed_1': 64,  'band_1': 65,
    'elf_1': 66,  'gap_1': 67,  'gas_1': 68,  'screen_1': 69,
    'raft_1': 70,  'flow_1': 71,  'cod_1': 72,  'switch_1': 73,
    'nerd_1': 74,  'spear_1': 75,  'crab_1': 76,  'work_1': 77,
    'bib_1': 78,  'pound_1': 79,  'stunt_1': 80,  'chin_1': 81,
    'mutt_2': 82,  'raft_2': 83,  'watch_2': 84,  'gull_2': 85,
    'hound_2': 86,  'gas_2': 87,  'dark_2': 88,  'loop_2': 89,
    'edge_2': 90,  'seat_2': 91,  'cod_2': 92,  'curb_2': 93,
    'guise_2': 94,  'flow_2': 95,  'verse_2': 96,  'tooth_2': 97,
    'bib_2': 98,  'sling_2': 99,  'band_2': 100,  'ramp_2': 101,
    'slang_2': 102,  'loom_2': 103,  'crab_2': 104,  'scalp_2': 105,
    'fluff_2': 106,  'stunt_2': 107,  'fog_2': 108,  'mulch_2': 109,
    'lobe_2': 110,  'elf_2': 111,  'fate_2': 112,  'chin_2': 113,
    'gig_2': 114,  'pump_2': 115,  'prince_2': 116,  'scotch_2': 117,
    'knee_2': 118,  'jive_2': 119,  'strap_2': 120,  'dude_2': 121,
    'road_2': 122,  'chore_2': 123,  'spear_2': 124,  'sketch_2': 125,
    'lime_2': 126,  'shed_2': 127,  'mall_2': 128,  'thumb_2': 129,
    'gap_2': 130,  'spleen_2': 131,  'switch_2': 132,  'crud_2': 133,
    'nerd_2': 134,  'ranch_2': 135,  'heel_2': 136,  'root_2': 137,
    'slope_2': 138,  'pile_2': 139,  'smirk_2': 140,  'pig_2': 141,
    'work_2': 142,  'graft_2': 143,  'map_2': 144,  'kit_2': 145,
    'screen_2': 146,  'sword_2': 147,  'feat_2': 148,  'chain_2': 149,
    'block_2': 150,  'pound_2': 151,  'valve_2': 152,  'plan_2': 153,
    'gap_3': 154,  'spleen_3': 155,  'tooth_3': 156,  'slang_3': 157,
    'thumb_3': 158,  'loom_3': 159,  'shed_3': 160,  'sword_3': 161,
    'lobe_3': 162,  'work_3': 163,  'slope_3': 164,  'mulch_3': 165,
    'cod_3': 166,  'block_3': 167,  'gig_3': 168,  'verse_3': 169,
    'edge_3': 170,  'sling_3': 171,  'gull_3': 172,  'plan_3': 173,
    'curb_3': 174,  'root_3': 175,  'mutt_3': 176,  'strap_3': 177,
    'road_3': 178,  'band_3': 179,  'spear_3': 180,  'pile_3': 181,
    'fog_3': 182,  'fate_3': 183,  'gas_3': 184,  'pig_3': 185,
    'crab_3': 186,  'flow_3': 187,  'bib_3': 188,  'sketch_3': 189,
    'raft_3': 190,  'ranch_3': 191,  'switch_3': 192,  'mall_3': 193,
    'kit_3': 194,  'heel_3': 195,  'hound_3': 196,  'map_3': 197,
    'crud_3': 198,  'screen_3': 199,  'pump_3': 200,  'loop_3': 201,
    'seat_3': 202,  'fluff_3': 203,  'feat_3': 204,  'valve_3': 205,
    'knee_3': 206,  'guise_3': 207,  'jive_3': 208,  'lime_3': 209,
    'chain_3': 210,  'graft_3': 211,  'pound_3': 212,  'dude_3': 213,
    'dark_3': 214,  'ramp_3': 215,  'chin_3': 216,  'scotch_3': 217,
    'elf_3': 218,  'nerd_3': 219,  'watch_3': 220,  'prince_3': 221,
    'smirk_3': 222,  'scalp_3': 223,  'chore_3': 224,  'stunt_3': 225}

event_id_res = {
    # extras
    'fixation': 1,  'start-phase': 2,  'stop-phase': 3,
    'r1-start': 230,  'r1-stop': 231,  'r2-start': 232,  'r2-stop': 233,
    'r3-start': 234,  'r3-stop': 235,  'r4-start': 236,  'r4-stop': 237,
    'r5-start': 238,  'r5-stop': 239,  'r6-start': 240,  'r6-stop': 241,
    # items
    'shop_1': 10,  'beef_1': 11,  'site_1': 12,  'neck_1': 13,
    'flask_1': 14,  'worm_1': 15,  'clown_1': 16,  'school_1': 17,
    'wing_1': 18,  'squad_1': 19,  'drake_1': 20,  'string_1': 21,
    'bulge_1': 22,  'jeep_1': 23,  'wire_1': 24,  'day_1': 25,
    'mound_1': 26,  'wrench_1': 27,  'bone_1': 28,  'rye_1': 29,
    'field_1': 30,  'heist_1': 31,  'ski_1': 32,  'coast_1': 33,
    'hot_1': 34,  'stuff_1': 35,  'pearl_1': 36,  'grail_1': 37,
    'guard_1': 38,  'spud_1': 39,  'fox_1': 40,  'poll_1': 41,
    'knock_1': 42,  'lip_1': 43,  'brace_1': 44,  'boat_1': 45,
    'ledge_1': 46,  'fleece_1': 47,  'vice_1': 48,  'job_1': 49,
    'haste_1': 50,  'freight_1': 51,  'vase_1': 52,  'hunk_1': 53,
    'rum_1': 54,  'range_1': 55,  'rift_1': 56,  'shape_1': 57,
    'quest_1': 58,  'past_1': 59,  'doll_1': 60,  'earl_1': 61,
    'lad_1': 62,  'snot_1': 63,  'ham_1': 64,  'scout_1': 65,
    'pie_1': 66,  'hive_1': 67,  'gym_1': 68,  'eye_1': 69,
    'shin_1': 70,  'flux_1': 71,  'scone_1': 72,  'frat_1': 73,
    'crib_1': 74,  'piece_1': 75,  'tech_1': 76,  'wand_1': 77,
    'slot_1': 78,  'fuzz_1': 79,  'path_1': 80,  'mite_1': 81,
    'shop_2': 82,  'beef_2': 83,  'site_2': 84,  'neck_2': 85,
    'flask_2': 86,  'worm_2': 87,  'clown_2': 88,  'school_2': 89,
    'wing_2': 90,  'squad_2': 91,  'drake_2': 92,  'string_2': 93,
    'bulge_2': 94,  'jeep_2': 95,  'wire_2': 96,  'day_2': 97,
    'mound_2': 98,  'wrench_2': 99,  'bone_2': 100,  'rye_2': 101,
    'field_2': 102,  'heist_2': 103,  'ski_2': 104,  'coast_2': 105,
    'hot_2': 106,  'stuff_2': 107,  'pearl_2': 108,  'grail_2': 109,
    'guard_2': 110,  'spud_2': 111,  'fox_2': 112,  'poll_2': 113,
    'knock_2': 114,  'lip_2': 115,  'brace_2': 116,  'boat_2': 117,
    'ledge_2': 118,  'fleece_2': 119,  'vice_2': 120,  'job_2': 121,
    'haste_2': 122,  'freight_2': 123,  'vase_2': 124,  'hunk_2': 125,
    'rum_2': 126,  'range_2': 127,  'rift_2': 128,  'shape_2': 129,
    'quest_2': 130,  'past_2': 131,  'doll_2': 132,  'earl_2': 133,
    'lad_2': 134,  'snot_2': 135,  'ham_2': 136,  'scout_2': 137,
    'pie_2': 138,  'hive_2': 139,  'gym_2': 140,  'eye_2': 141,
    'shin_2': 142,  'flux_2': 143,  'scone_2': 144,  'frat_2': 145,
    'crib_2': 146,  'piece_2': 147,  'tech_2': 148,  'wand_2': 149,
    'slot_2': 150,  'fuzz_2': 151,  'path_2': 152,  'mite_2': 153,
    'shop_3': 154,  'beef_3': 155,  'site_3': 156,  'neck_3': 157,
    'flask_3': 158,  'worm_3': 159,  'clown_3': 160,  'school_3': 161,
    'wing_3': 162,  'squad_3': 163,  'drake_3': 164,  'string_3': 165,
    'bulge_3': 166,  'jeep_3': 167,  'wire_3': 168,  'day_3': 169,
    'mound_3': 170,  'wrench_3': 171,  'bone_3': 172,  'rye_3': 173,
    'field_3': 174,  'heist_3': 175,  'ski_3': 176,  'coast_3': 177,
    'hot_3': 178,  'stuff_3': 179,  'pearl_3': 180,  'grail_3': 181,
    'guard_3': 182,  'spud_3': 183,  'fox_3': 184,  'poll_3': 185,
    'knock_3': 186,  'lip_3': 187,  'brace_3': 188,  'boat_3': 189,
    'ledge_3': 190,  'fleece_3': 191,  'vice_3': 192,  'job_3': 193,
    'haste_3': 194,  'freight_3': 195,  'vase_3': 196,  'hunk_3': 197,
    'rum_3': 198,  'range_3': 199,  'rift_3': 200,  'shape_3': 201,
    'quest_3': 202,  'past_3': 203,  'doll_3': 204,  'earl_3': 205,
    'lad_3': 206,  'snot_3': 207,  'ham_3': 208,  'scout_3': 209,
    'pie_3': 210,  'hive_3': 211,  'gym_3': 212,  'eye_3': 213,
    'shin_3': 214,  'flux_3': 215,  'scone_3': 216,  'frat_3': 217,
    'crib_3': 218,  'piece_3': 219,  'tech_3': 220,  'wand_3': 221,
    'slot_3': 222,  'fuzz_3': 223,  'path_3': 224,  'mite_3': 225}

windows = {"Prestim": (-200, -0.5),
           "Event": (0, 4000)
           }

# Make sure that the needed folders exists. Create if not.

folder_list = ['sigma\\butterfly_plots', 'sigma\\topoplots', 'sigma\\psd_plots', 'sigma\\erp_plots',
               'sigma\\processed', 'sigma\\ica']

for l in folder_list:
    if not os.path.exists(l):
        os.makedirs(l)

# loop through the list of raw files for preprocessing
for f in raw_files:

    # extract participant number and condition
    s_number = '_'.join(f.split('_')[:1])
    condition = '_'.join(f.split('_')[3:])
    condition = '_'.join(condition.split('.')[:1])
    s_no_cond = s_number+condition

    # ---------------------------------------------------------------------------
    # Basic Pre-Processing and ICA Correction
    # ---------------------------------------------------------------------------

    print('processing participant: ' + s_number + ". condition:" + condition)

    if op.exists('sigma\\processed\\' + s_number + "_" + condition + '_epo.fif.gz'):
        if not(compute_from_scratch):
            print('skipping participant' + s_number + ': file already exists')
            continue

    raw = mne.io.read_raw_brainvision(f, preload=True, eog=['E1', 'E2'],
                                      misc=['EMG1', 'EMG2', 'EMG3', 'ECG'])

    # Fix files
    # pasting files together in cases of a crash

    if f == '07_dnap_int_ret.vhdr':
        raw1 = mne.io.read_raw_brainvision('07_dnap_int_ret.vhdr', preload=True, eog=['E1', 'E2'],
                                           misc=['EMG1', 'EMG2', 'EMG3', 'ECG'])
        raw2 = mne.io.read_raw_brainvision('files_to_paste\\07_dnap_int_ret2.vhdr', preload=True, eog=['E1', 'E2'],
                                           misc=['EMG1', 'EMG2', 'EMG3', 'ECG'])
        raws = [raw1, raw2]
        raw = mne.io.concatenate_raws(raws, preload=True)
    elif f == '14_dnap_int_res.vhdr':
        raw1 = mne.io.read_raw_brainvision('14_dnap_int_res.vhdr', preload=True, eog=['E1', 'E2'],
                                           misc=['EMG1', 'EMG2', 'EMG3', 'ECG'])
        raw2 = mne.io.read_raw_brainvision('files_to_paste\\14_dnap_int_res2.vhdr', preload=True, eog=['E1', 'E2'],
                                           misc=['EMG1', 'EMG2', 'EMG3', 'ECG'])
        raws = [raw1, raw2]
        raw = mne.io.concatenate_raws(raws, preload=True)
    elif f == '16_dnap_int_ret.vhdr':
        raw1 = mne.io.read_raw_brainvision('16_dnap_int_ret.vhdr', preload=True, eog=['E1', 'E2'],
                                           misc=['EMG1', 'EMG2', 'EMG3', 'ECG'])
        raw2 = mne.io.read_raw_brainvision('files_to_paste\\16_dnap_int2_ret.vhdr', preload=True, eog=['E1', 'E2'],
                                           misc=['EMG1', 'EMG2', 'EMG3', 'ECG'])
        raws = [raw1, raw2]
        raw = mne.io.concatenate_raws(raws, preload=True)
    elif f == '19_dnap_int_res.vhdr':
        raw1 = mne.io.read_raw_brainvision('19_dnap_int_res.vhdr', preload=True, eog=['E1', 'E2'],
                                           misc=['EMG1', 'EMG2', 'EMG3', 'ECG'])
        raw2 = mne.io.read_raw_brainvision('files_to_paste\\19_dnap_int_res2.vhdr', preload=True, eog=['E1', 'E2'],
                                           misc=['EMG1', 'EMG2', 'EMG3', 'ECG'])
        raws = [raw1, raw2]
        raw = mne.io.concatenate_raws(raws, preload=True)
    elif f == '25_dnap_int_ret.vhdr':
        raw1 = mne.io.read_raw_brainvision('25_dnap_int_ret.vhdr', preload=True, eog=['E1', 'E2'],
                                           misc=['EMG1', 'EMG2', 'EMG3', 'ECG'])
        raw2 = mne.io.read_raw_brainvision('files_to_paste\\25_dnap_int_ret2.vhdr', preload=True, eog=['E1', 'E2'],
                                           misc=['EMG1', 'EMG2', 'EMG3', 'ECG'])
        raws = [raw1, raw2]
        raw = mne.io.concatenate_raws(raws, preload=True)
    else:
        pass

    # segment the continuous EEG around events
    events = mne.events_from_annotations(raw)

    # crop out breaks between restudy and retrieval rounds
    events_array = events[0]

    sf = raw.info['sfreq']
    
    # set some cases manually if they were concatenated prior as time info is incorrect
    if f == '07_dnap_int_ret.vhdr':
        tmin1 = 106
        tmax1 = 593
        tmin2 = 1348
        tmax2 = 1813
        tmin3 = 2374
        tmax3 = 2809
        tmin4 = 3564
        tmax4 = 3991
        tmin5 = 4776
        tmax5 = 5184
        tmin6 = 5944
        tmax6 = 6348
    elif f == '14_dnap_int_res.vhdr':
        tmin1 = 37
        tmax1 = 262
        tmin2 = 1263
        tmax2 = 1486
        tmin3 = 2462
        tmax3 = 2678
        tmin4 = 3286
        tmax4 = 3507
        tmin5 = 4495
        tmax5 = 4715
        tmin6 = 5698
        tmax6 = 5916
    elif f == '16_dnap_int_ret.vhdr':
        tmin1 = 61
        tmax1 = 529
        tmin2 = 651
        tmax2 = 1105
        tmin3 = 1854
        tmax3 = 2307
        tmin4 = 3053
        tmax4 = 3509
        tmin5 = 4318
        tmax5 = 4754
        tmin6 = 5440
        tmax6 = 5904
    elif f == '19_dnap_int_res.vhdr':
        tmin1 = 59
        tmax1 = 286
        tmin2 = 995
        tmax2 = 1214
        tmin3 = 2214
        tmax3 = 2438
        tmin4 = 3408
        tmax4 = 3628
        tmin5 = 4546
        tmax5 = 4767
        tmin6 = 5823
        tmax6 = 6041
    elif f == '25_dnap_int_ret.vhdr':
        tmin1 = 35
        tmax1 = 493
        tmin2 = 1113
        tmax2 = 1512
        tmin3 = 1952
        tmax3 = 2338
        tmin4 = 3162
        tmax4 = 3552
        tmin5 = 4448
        tmax5 = 4832
        tmin6 = 5559
        tmax6 = 5959
    else:
        for trigger in range(0, len(events_array)):
            trig_all = events_array[trigger, 2]
            if trig_all == 230:
                tmin1_time = events_array[trigger, 0]/sf
                tmin1 = tmin1_time - 0.01
                print("tmin1 =", tmin1)
            elif trig_all == 231:
                tmax1_time = events_array[trigger, 0]/sf
                tmax1 = tmax1_time + 0.01
                print("tmax1 =", tmax1)
            elif trig_all == 232:
                tmin2_time = events_array[trigger, 0]/sf
                tmin2 = tmin2_time - 0.01
                print("tmin2 =", tmin2)
            elif trig_all == 233:
                tmax2_time = events_array[trigger, 0]/sf
                tmax2 = tmax2_time + 0.01
                print("tmax2 =", tmax2)
            elif trig_all == 234:
                tmin3_time = events_array[trigger, 0]/sf
                tmin3 = tmin3_time - 0.01
                print("tmin3 =", tmin3)
            elif trig_all == 235:
                tmax3_time = events_array[trigger, 0]/sf
                tmax3 = tmax3_time + 0.01
                print("tmax3 =", tmax3)
            elif trig_all == 236:
                tmin4_time = events_array[trigger, 0]/sf
                tmin4 = tmin4_time - 0.01
                print("tmin4 =", tmin4)
            elif trig_all == 237:
                tmax4_time = events_array[trigger, 0]/sf
                tmax4 = tmax4_time + 0.01
                print("tmax4 =", tmax4)
            elif trig_all == 238:
                tmin5_time = events_array[trigger, 0]/sf
                tmin5 = tmin5_time - 0.01
                print("tmin5 =", tmin5)
            elif trig_all == 239:
                tmax5_time = events_array[trigger, 0]/sf
                tmax5 = tmax5_time + 0.01
                print("tmax5 =", tmax5)
            elif trig_all == 240:
                tmin6_time = events_array[trigger, 0]/sf
                tmin6 = tmin6_time - 0.01
                print("tmin6 =", tmin6)
            elif trig_all == 241:
                tmax6_time = events_array[trigger, 0]/sf
                tmax6 = tmax6_time + 0.01
                print("tmax6 =", tmax6)
    raw2 = raw.copy()
    section1 = raw2.crop(tmin=tmin1, tmax=tmax1,
                         include_tmax=True, verbose=None)
    raw2 = raw.copy()
    section2 = raw2.crop(tmin=tmin2, tmax=tmax2,
                         include_tmax=True, verbose=None)
    raw2 = raw.copy()
    section3 = raw2.crop(tmin=tmin3, tmax=tmax3,
                         include_tmax=True, verbose=None)
    raw2 = raw.copy()
    section4 = raw2.crop(tmin=tmin4, tmax=tmax4,
                         include_tmax=True, verbose=None)
    raw2 = raw.copy()
    section5 = raw2.crop(tmin=tmin5, tmax=tmax5,
                         include_tmax=True, verbose=None)
    raw2 = raw.copy()
    section6 = raw2.crop(tmin=tmin6, tmax=tmax6,
                         include_tmax=True, verbose=None)

    raws = [section1, section2, section3, section4, section5, section6]

    raw = mne.concatenate_raws(
        raws, preload=None, events_list=None, on_mismatch='raise', verbose=None)

    # downsample to 250 Hz
    raw = raw.resample(100)

    # re-reference to the mean of the left and right mastoids
    raw.set_eeg_reference(ref_channels=['M1', 'M2'])

    # drop temporal channels because they are noisy
    raw.drop_channels(['M1', 'M2', 'EMG1', 'EMG2', 'EMG3', 'ECG'])

    # set montage to add information about electrode positions
    raw.set_montage(montage)

    # apply ICA-based EOG correction

    # filter the data
    raw.filter(1, 40., l_trans_bandwidth='auto', h_trans_bandwidth='auto',
               filter_length='auto', method='fir', fir_window='hamming', phase='zero', n_jobs=2)

    # Create initial PSD plot
    psd_plot_name = 'sigma\\psd_plots\\' + s_number + \
        "_" + condition + '_psd_raw' + '.png'
    fig = raw.plot_psd(fmin=0, fmax=30)
    fig.savefig(psd_plot_name)
    plt.close(fig='all')

    # apply ICA function from utils.py script
    raw = compute_ica_correction(raw, f)

    picks = mne.pick_types(raw.info, eeg=True, eog=False,
                           stim=False, misc=False)

    epochs = mne.make_fixed_length_epochs(raw, duration=30,preload=True)

    # save preprocessed data
    processed_file = 'sigma\\processed\\' + \
        s_number + "_" + condition + '_epo.fif.gz'
    epochs.save(processed_file, fmt='single', overwrite=True)
