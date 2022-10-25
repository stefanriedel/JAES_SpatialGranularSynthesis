import json
from os.path import dirname, join as pjoin
import numpy as np
import os
import pandas as pd
from Utility.pairwiseTests import posthoc_wilcoxon


root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'ExperimentData', 'exp2')
save_dir = pjoin(root_dir, 'ExperimentData', 'pvalues_tables')

file_list = os.listdir(data_dir)

if '.DS_Store' in file_list:
    file_list.remove('.DS_Store')

N = len(file_list)

# lev = listener envelopment
# leg = listener engulfment
all_ratings_lev = np.zeros((N, 8, 4))
all_ratings_leg = np.zeros((N, 8, 4))

for subj in range(N):
    f_location = pjoin(data_dir, file_list[subj])

    f = open(f_location)

    data = json.load(f)

    results = data['Results']
    parts = results['Parts']

    part_lev = parts[0]
    part_leg = parts[1]
    trials_lev = part_lev['Trials']
    trials_leg = part_leg['Trials']

    for idx in range(len(trials_lev)):
        all_ratings_lev[subj, :, idx] = trials_lev[idx]['Ratings']
        all_ratings_leg[subj, :, idx] = trials_leg[idx]['Ratings']

# Write to semantic trial variables for easier plotting

median_all_ratings_lev = np.median(all_ratings_lev, axis=0)
median_all_ratings_leg = np.median(all_ratings_leg, axis=0)

# --------------------------------------- DATA DICTIONARY ---------------------------------------
# structure of data: N x 8 x 4
# N participants
# 8 stimuli x 4 trials
#
# trials: 0 (layers pink), 1 (bw pink), 2 (layers concret ph), 3 (bw concret ph)
#
# stimuli trial 1 and 3: 0,1,2 (broadband L1,L2L3,L3), 3,4,5 (lowpass L1,L2L3,L3), 6 (stereo), 7 (vog)
# stimuli trial 0 and 2: 0,1,2,3,4,5 (L1, L1L2, L2, L2L3, L3, L1L2L3), 6 (stereo), 7 (vog)

ratings_lev_layers_pink = all_ratings_lev[:,:,0]
ratings_lev_layers_concret = all_ratings_lev[:,:,2]
ratings_leg_layers_pink = all_ratings_leg[:,:,0]
ratings_leg_layers_concret = all_ratings_leg[:,:,2]

ratings_lev_bw_pink = all_ratings_lev[:,:,1]
ratings_lev_bw_concret = all_ratings_lev[:,:,3]
ratings_leg_bw_pink = all_ratings_leg[:,:,1]
ratings_leg_bw_concret = all_ratings_leg[:,:,3]

if True:
    data_stack = np.array([ratings_lev_layers_pink[:,:].T , ratings_lev_layers_concret[:,:].T])
    pairs_to_be_tested = [[0,3], [0,4]]
    #labels_X = ['L1L2', 'L2', 'L2L3', 'L3', 'L1L2L3']
    labels_X = ['L2L3', 'L3']
    labels_Y = ['L1 (Pink)', 'L1 (ConcretPH)']
    pvals_array = np.zeros((2,2))
    idx = 0
    for sound in range(2):
        trial_ratings = data_stack[sound,:,:]
        ratings = trial_ratings[:,:]
        #pvals = posthoc_wilcoxon(ratings, pairs_to_be_tested)
        pvals_array[idx, :] = posthoc_wilcoxon(ratings, pairs_to_be_tested)
        idx += 1

    pvals = pd.DataFrame(pvals_array)
    pvals = pvals.round(decimals=3)
    pvals = pvals.set_axis(labels_Y, axis=0)
    pvals = pvals.set_axis(labels_X, axis=1)

    table_tex = pvals.to_latex()
    tex_file = open("ExperimentData/pvalue_tables/" + "BH_corrected_EXP2_EnvelopmentLayers" + ".tex", "w")
    tex_file.write(table_tex)
    tex_file.close()

if True:
    data_stack = np.array([ratings_leg_layers_pink[:,:].T , ratings_leg_layers_concret[:,:].T])
    pairs_to_be_tested = [[0,3], [0,4]]
    labels_X = ['L2L3', 'L3']
    labels_Y = ['L1 (Pink)', 'L1 (ConcretPH)']
    pvals_array = np.zeros((2,2))
    idx = 0
    for sound in range(2):
        trial_ratings = data_stack[sound,:,:]
        ratings = trial_ratings[:,:]
        #pvals = posthoc_wilcoxon(ratings, pairs_to_be_tested)
        pvals_array[idx, :] = posthoc_wilcoxon(ratings, pairs_to_be_tested)
        idx += 1

    pvals = pd.DataFrame(pvals_array)
    pvals = pvals.round(decimals=3)
    pvals = pvals.set_axis(labels_Y, axis=0)
    pvals = pvals.set_axis(labels_X, axis=1)

    table_tex = pvals.to_latex()
    tex_file = open("ExperimentData/pvalue_tables/" + "BH_corrected_EXP2_EngulfmentLayers" + ".tex", "w")
    tex_file.write(table_tex)
    tex_file.close()

if True:
    data_stack = np.array([ratings_lev_bw_pink[:,:].T , ratings_lev_bw_concret[:,:].T])
    pairs_to_be_tested = [[0,1], [0,2]]
    labels_X = ['L2L3', 'L3']
    labels_Y = ['L1 (Pink Broadband)', 'L1 (Pink Lowpass)', 'L1 (ConcretPH Broadband)', 'L1 (ConcretPH Lowpass)']
    pvals_array = np.zeros((4,2))
    idx = 0
    for sound in range(2):
        for bw in range(2):
            trial_ratings = data_stack[sound,:,:]
            ratings = trial_ratings[bw*3:(bw+1)*3,:]
            #pvals = posthoc_wilcoxon(ratings, pairs_to_be_tested)
            pvals_array[idx, :] = posthoc_wilcoxon(ratings, pairs_to_be_tested)
            idx += 1

    pvals = pd.DataFrame(pvals_array)
    pvals = pvals.round(decimals=3)
    pvals = pvals.set_axis(labels_Y, axis=0)
    pvals = pvals.set_axis(labels_X, axis=1)

    table_tex = pvals.to_latex()
    tex_file = open("ExperimentData/pvalue_tables/" + "BH_corrected_EXP2_EnvelopmentBandwidths" + ".tex", "w")
    tex_file.write(table_tex)
    tex_file.close()

if True:
    data_stack = np.array([ratings_leg_bw_pink[:,:].T , ratings_leg_bw_concret[:,:].T])
    pairs_to_be_tested = [[0,1], [0,2]]
    labels_X = ['L2L3', 'L3']
    labels_Y = ['L1 (Pink Broadband)', 'L1 (Pink Lowpass)', 'L1 (ConcretPH Broadband)', 'L1 (ConcretPH Lowpass)']
    pvals_array = np.zeros((4,2))
    idx = 0
    for sound in range(2):
        for bw in range(2):
            trial_ratings = data_stack[sound,:,:]
            ratings = trial_ratings[bw*3:(bw+1)*3,:]
            #pvals = posthoc_wilcoxon(ratings, pairs_to_be_tested)
            pvals_array[idx, :] = posthoc_wilcoxon(ratings, pairs_to_be_tested)
            idx += 1

    pvals = pd.DataFrame(pvals_array)
    pvals = pvals.round(decimals=3)
    pvals = pvals.set_axis(labels_Y, axis=0)
    pvals = pvals.set_axis(labels_X, axis=1)

    table_tex = pvals.to_latex()
    tex_file = open("ExperimentData/pvalue_tables/" + "BH_corrected_EXP2_EngulfmentBandwidths" + ".tex", "w")
    tex_file.write(table_tex)
    tex_file.close()

print('done')

