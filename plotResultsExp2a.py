import json
from os.path import dirname, join as pjoin
import numpy as np
import matplotlib.pyplot as plt
import os
from Utility.plotsLookAndFeel import *

BLOCK_PLT_SHOW = False
format = '.eps'

root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'ExperimentData', 'exp2')
figures_dir = pjoin(root_dir, 'Figures', 'ExperimentResults')

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

iqr_ratings_lev = np.percentile(all_ratings_lev, [25, 75], axis=0)
asymmetric_err_lev = np.array([
    median_all_ratings_lev - iqr_ratings_lev[0, :, :],
    iqr_ratings_lev[1, :, :] - median_all_ratings_lev
])

iqr_ratings_leg = np.percentile(all_ratings_leg, [25, 75], axis=0)
asymmetric_err_leg = np.array([
    median_all_ratings_leg - iqr_ratings_leg[0, :, :],
    iqr_ratings_leg[1, :, :] - median_all_ratings_leg
])

# LEV&LEG vs. layers
labels_list_layers = [['Envelopment', 'Engulfment'],
                      ['Envelopment', 'Engulfment']]
titles_list = ['Pink noise: ' + r'$\Delta t = 5$' + ' ms,  '  + r'$L = 250$' + ' ms', 'Concret PH: ' + r'$\Delta t = 5$' + ' ms,  '  + r'$L = 250$' + ' ms']

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6 * 2, 3), sharey=True, gridspec_kw = {'wspace':0.05, 'hspace':0})
xaxis = np.arange(0, 8)
for idx in range(2):
    markersz = 8
    linest = ':'
    style_width = 1.5

    axs[idx].plot(xaxis - 0.1, median_all_ratings_lev[:, idx * 2], label=labels_list_layers[idx][0], marker=lev_mkr, markersize=markersz, markerfacecolor=mkr_color_lev,
                  markeredgecolor='k', linestyle=lev_linest, linewidth=style_width, color=linecolor, zorder=3)
    axs[idx].errorbar(xaxis - 0.1,y=median_all_ratings_lev[:, idx * 2], capsize=4.0, linestyle='none', linewidth=style_width, xerr=0,
                      yerr=asymmetric_err_lev[:, :, idx * 2], color='k', zorder=2)

    axs[idx].plot(xaxis + 0.1, median_all_ratings_leg[:, idx * 2], label=labels_list_layers[idx][1], marker=leg_mkr, markersize=markersz, markerfacecolor=mkr_color_leg,
                  markeredgecolor='k', linestyle=leg_linest, linewidth=style_width, color=linecolor, zorder=3)
    axs[idx].errorbar(xaxis + 0.1, y=median_all_ratings_leg[:, idx * 2], capsize=4.0, linestyle='none', linewidth=style_width, xerr=0,
                      yerr=asymmetric_err_leg[:, :, idx * 2], color='k', zorder=2)

    axs[idx].set_ylim(-5, 105)
    axs[idx].set_xticks(xaxis)
    axs[idx].set_xticklabels(['L1', 'L1L2', 'L2', 'L2L3', 'L3', 'L1L2L3', 'SP', 'ZEN'])
    axs[idx].set_xlabel('Active loudspeaker set')
    axs[idx].set_yticks([0, 25, 50, 75, 100])
    if idx==0:
        axs[idx].set_ylabel('Envelopment / Engulfment')
    axs[idx].set_title(titles_list[idx])
    axs[idx].grid()
    axs[idx].legend(framealpha=1)
plt.savefig(fname=pjoin(figures_dir, 'layers_lev_leg' + format),
            bbox_inches='tight')
plt.show(block=BLOCK_PLT_SHOW)
