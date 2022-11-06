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

ratings_leg_bw_pink = all_ratings_leg[:,:,1]
ratings_leg_bw_concret = all_ratings_leg[:,:,3]

ratings_lev_bw_pink = all_ratings_lev[:,:,1]
ratings_lev_bw_concret = all_ratings_lev[:,:,3]


iqr_ratings_lev = np.percentile(all_ratings_lev, [25, 75], axis=0)
asymmetric_iqr_lev = np.array([
    median_all_ratings_lev - iqr_ratings_lev[0, :, :],
    iqr_ratings_lev[1, :, :] - median_all_ratings_lev
])

iqr_ratings_leg = np.percentile(all_ratings_leg, [25, 75], axis=0)
asymmetric_iqr_leg = np.array([
    median_all_ratings_leg - iqr_ratings_leg[0, :, :],
    iqr_ratings_leg[1, :, :] - median_all_ratings_leg
])

# LEV or LEG vs. layers and bandwidth
labels_list_bw = [['Broadband', 'Lowpass'],
                  ['Broadband', 'Lowpass'],
                  ['Broadband', 'Lowpass'],
                  ['Broadband', 'Lowpass']]

savename_list_bw_lev = ['pink_bw_lev.pdf', 'concretph_bw_lev.pdf']

title_list = ['Pink noise: ' + r'$\Delta t = 5$' + ' ms,  '  + r'$L = 250$' + ' ms', 'Concret PH: ' + r'$\Delta t = 5$' + ' ms,  '  + r'$L = 250$' + ' ms']


xaxis = np.arange(0, 5)
data_idx_list = [0, 1, 0, 1]

scale = 3
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4*scale, scale), sharey=True, gridspec_kw = {'wspace':0.05, 'hspace':0})
# LEV
for ax_idx in range(2):
    idx = data_idx_list[ax_idx]

    markersz = 8
    linest = ':'

    # lev lowpass
    axs[ax_idx].plot(xaxis[:3] + 0.1, median_all_ratings_lev[3:6, idx * 2 + 1], label=labels_list_bw[idx][1], marker=lev_mkr, markersize=markersz, 
                markerfacecolor=lowpass_color, linestyle=lowpass_linest, color=linecolor, markeredgecolor='k', zorder=3)
    axs[ax_idx].errorbar(xaxis[:3] + 0.1, y=median_all_ratings_lev[3:6, idx * 2 + 1], capsize=4.0, linestyle='none', 
                xerr=0, yerr=asymmetric_iqr_lev[:, 3:6, idx * 2 + 1], color='k', zorder=2)

    # lev broadband
    axs[ax_idx].plot(xaxis[:3] - 0.1, median_all_ratings_lev[:3, idx * 2 + 1], label=labels_list_bw[idx][0], marker=lev_mkr, markersize=markersz, 
                markerfacecolor=mkr_color_lev, linestyle=lev_linest, color=linecolor, markeredgecolor='k', zorder=3)
    axs[ax_idx].errorbar(xaxis[:3] - 0.1, y=median_all_ratings_lev[:3, idx * 2 + 1], capsize=4.0, linestyle='none', 
                xerr=0, yerr=asymmetric_iqr_lev[:, :3, idx * 2 + 1], color='k', zorder=2)

    # lev stereo vog
    axs[ax_idx].plot(xaxis[3:5] - 0.1, median_all_ratings_lev[6:8, idx * 2 + 1], marker=lev_mkr, markersize=markersz, 
                markerfacecolor=mkr_color_lev, linestyle=lev_linest, color=linecolor, markeredgecolor='k', zorder=3)
    axs[ax_idx].errorbar(xaxis[3:5] - 0.1, y=median_all_ratings_lev[6:8, idx * 2 + 1], capsize=4.0, 
                linestyle='none', xerr=0, yerr=asymmetric_iqr_lev[:, 6:8, idx * 2 + 1], color='k', zorder=2)

    axs[ax_idx].set_ylim(-5, 105)
    axs[ax_idx].set_xticks(xaxis)
    axs[ax_idx].set_xticklabels(['L1', 'L2L3', 'L3', 'SP', 'ZEN'])
    axs[ax_idx].set_xlabel('Active loudspeaker set')
    axs[ax_idx].set_yticks([0, 25, 50, 75, 100])
    if ax_idx == 0:
        axs[ax_idx].set_ylabel('Envelopment')
    axs[ax_idx].set_title(title_list[ax_idx])
    axs[ax_idx].grid()
    axs[ax_idx].legend(framealpha=1)
plt.savefig(fname=pjoin(figures_dir, 'bandwidths_lev' + format),
        bbox_inches='tight')
plt.show(block=BLOCK_PLT_SHOW)


# PLOT bandwidth effect on engulfment
scale = 3
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4*scale, scale), sharey=True, gridspec_kw = {'wspace':0.05, 'hspace':0})
# LEG
for ax_idx in range(2):
    idx = data_idx_list[ax_idx]

    markersz = 8
    linest = ':'

    # leg lowpass
    #axs[ax_idx].plot(xaxis[:3] + 0.1, median_all_ratings_leg[3:6, idx * 2 + 1], linestyle=leg_linest, color='gray', zorder=1)
    axs[ax_idx].plot(xaxis[:3] + 0.1, median_all_ratings_leg[3:6, idx * 2 + 1], label=labels_list_bw[idx][1], marker=leg_mkr, markersize=markersz, 
                markerfacecolor=lowpass_color, linestyle=lowpass_linest, color=linecolor, markeredgecolor='k', zorder=3)
    axs[ax_idx].errorbar(xaxis[:3] + 0.1, y=median_all_ratings_leg[3:6, idx * 2 + 1], capsize=4.0, linestyle='none', xerr=0, yerr=asymmetric_iqr_leg[:, 3:6, idx * 2 + 1], color='k', zorder=2)

    # leg broadband
    #axs[ax_idx].plot(xaxis[:3] - 0.1, median_all_ratings_leg[:3, idx * 2 + 1], linestyle=leg_linest, color='gray', zorder=1)
    axs[ax_idx].plot(xaxis[:3] - 0.1, median_all_ratings_leg[:3, idx * 2 + 1], label=labels_list_bw[idx][0], marker=leg_mkr, markersize=markersz, 
                markerfacecolor=mkr_color_leg, linestyle=leg_linest, color=linecolor, markeredgecolor='k', zorder=3)
    axs[ax_idx].errorbar(xaxis[:3] - 0.1, y=median_all_ratings_leg[:3, idx * 2 + 1], capsize=4.0, linestyle='none', xerr=0, yerr=asymmetric_iqr_leg[:, :3, idx * 2 + 1], color='k', zorder=2)

    # leg stereo vog
    axs[ax_idx].plot(xaxis[3:5] - 0.1, median_all_ratings_leg[6:8, idx * 2 + 1], marker=leg_mkr, markersize=markersz, 
                markerfacecolor=mkr_color_leg, linestyle=leg_linest, color=linecolor, markeredgecolor='k', zorder=3)
    axs[ax_idx].errorbar(xaxis[3:5] - 0.1, y=median_all_ratings_leg[6:8, idx * 2 + 1], capsize=4.0, linestyle='none', xerr=0, yerr=asymmetric_iqr_leg[:, 6:8, idx * 2 + 1], color='k', zorder=2)

    axs[ax_idx].set_ylim(-5, 105)
    axs[ax_idx].set_xticks(xaxis)
    axs[ax_idx].set_xticklabels(['L1', 'L2L3', 'L3', 'SP', 'ZEN'])
    axs[ax_idx].set_xlabel('Active loudspeaker set')
    axs[ax_idx].set_yticks([0, 25, 50, 75, 100])
    if ax_idx == 0:
        axs[ax_idx].set_ylabel('Engulfment')
    axs[ax_idx].set_title(title_list[ax_idx])
    axs[ax_idx].grid()
    axs[ax_idx].legend(framealpha=1)
plt.savefig(fname=pjoin(figures_dir, 'bandwidths_leg' + format),
        bbox_inches='tight')
plt.show(block=BLOCK_PLT_SHOW)
