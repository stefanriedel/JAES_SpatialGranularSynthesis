import json
from os.path import dirname, join as pjoin
import os
import numpy as np
import matplotlib.pyplot as plt
from Utility.pairwiseTests import posthoc_wilcoxon
import pandas as pd

root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'ExperimentData', 'exp1')
save_dir = pjoin(root_dir, 'ExperimentData', 'pvalues_tables')

file_list = os.listdir(data_dir)

if '.DS_Store' in file_list:
    file_list.remove('.DS_Store')
if 'stats' in file_list:
    file_list.remove('stats')

N = len(file_list)

all_ratings = np.zeros((N,8,6))

for subj in range(N):
    f_location = pjoin(data_dir, file_list[subj])

    f = open(f_location)

    data = json.load(f)

    results = data['Results']
    parts = results['Parts']

    part = parts[0]
    trials = part['Trials']


    for idx in range(len(trials)):
        all_ratings[subj,:,idx] = trials[idx]['Ratings']

order = [1,0,3,2]

ratings_pink0pt5ms = all_ratings[:,:,order[0]]
ratings_pink250ms = all_ratings[:,:,order[1]]
ratings_vocal0pt5ms = all_ratings[:,:,order[2]]
ratings_vocal250ms = all_ratings[:,:,order[3]]

ratings_bandwidth = all_ratings[:,:,5].T

data_stack = np.array([ratings_pink0pt5ms[:,:].T , ratings_pink250ms[:,:].T, ratings_vocal0pt5ms[:,:].T , ratings_vocal250ms[:,:].T])

if True:
    pairs_to_be_tested = [[0,1], [1,2], [2,3]]
    labels_X = ['100 vs. 20', '20 vs. 5', '5 vs. 1']
    labels_Y = ['T1 2D', 'T2 2D', 'T3 2D', 'T4 2D'] + ['T1 3D', 'T2 3D', 'T3 3D', 'T4 3D']
    pvals_array = np.zeros((8,3))
    idx = 0
    for spat in range(2):
        for trial in range(4):
            trial_ratings = data_stack[trial,:,:]
            ratings = trial_ratings[spat*4:(spat+1)*4,:]
            #pvals = posthoc_wilcoxon(ratings, pairs_to_be_tested)
            pvals_array[idx, :] = posthoc_wilcoxon(ratings, pairs_to_be_tested)
            idx += 1

    pvals = pd.DataFrame(pvals_array)
    pvals = pvals.round(decimals=3)
    pvals = pvals.set_axis(labels_Y, axis=0)
    pvals = pvals.set_axis(labels_X, axis=1)

    table_tex = pvals.to_latex()
    tex_file = open("ExperimentData/pvalue_tables/" + "BH_corrected_EXP1_clipped" + ".tex", "w")
    tex_file.write(table_tex)
    tex_file.close()

if True:
    pairs_to_be_tested = [[0,4], [1,5], [2,6], [3,7]]
    labels_X = ['100', '20', '5', '1']
    labels_Y = ['T1', 'T2', 'T3', 'T4']
    pvals_array = np.zeros((4,4))
    cliffs_delta = np.zeros((4,4))
    idx = 0
    for trial in range(4):
        trial_ratings = data_stack[trial,:,:]
        ratings = trial_ratings
        #pvals = posthoc_wilcoxon(ratings, pairs_to_be_tested)
        pvals_array[idx, :] = posthoc_wilcoxon(ratings, pairs_to_be_tested)
        idx += 1

    pvals = pd.DataFrame(pvals_array)
    pvals = pvals.round(decimals=3)
    pvals = pvals.set_axis(labels_Y, axis=0)
    pvals = pvals.set_axis(labels_X, axis=1)

    table_tex = pvals.to_latex()
    tex_file = open("ExperimentData/pvalue_tables/" + "BH_corrected_EXP1_SPAT_clipped" + ".tex", "w")
    tex_file.write(table_tex)
    tex_file.close()

if True:
    pairs_to_be_tested = [[0,1], [1,2], [2,3]]
    labels_X = ['SP vs. QP', 'QP vs L1', 'L1 vs. L1L2L3']
    labels_Y = ['Lowpass', 'Broadband']
    pvals_array = np.zeros((2,3))
    idx = 0
    for bw in range(2):
        ratings = ratings_bandwidth[bw*4:(bw+1)*4,:]
        pvals_array[idx, :] = posthoc_wilcoxon(ratings, pairs_to_be_tested)
        idx += 1

    pvals = pd.DataFrame(pvals_array)
    pvals = pvals.round(decimals=3)
    pvals = pvals.set_axis(labels_Y, axis=0)
    pvals = pvals.set_axis(labels_X, axis=1)

    table_tex = pvals.to_latex()
    tex_file = open("ExperimentData/pvalue_tables/" + "BH_corrected_EXP1_BW_clipped" + ".tex", "w")
    tex_file.write(table_tex)
    tex_file.close()


