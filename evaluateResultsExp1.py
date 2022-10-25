import json
from os.path import dirname, join as pjoin
import os
import numpy as np
import matplotlib.pyplot as plt
from Utility.plotsLookAndFeel import *

BLOCK_PLT_SHOW = False

root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'ExperimentData', 'exp1')
figures_dir = pjoin(root_dir, 'Figures', 'ExperimentResults')

file_list = os.listdir(data_dir)

if '.DS_Store' in file_list:
    file_list.remove('.DS_Store')

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

median_all_ratings = np.median(all_ratings, axis=0)
irq_all_ratings = np.percentile(all_ratings, [25,75], axis=0)
asymmetric_iqr = np.array([ median_all_ratings-irq_all_ratings[0,:,:] ,  irq_all_ratings[1,:,:]-median_all_ratings  ])

label_list = [['2D (L1)', '3D (L1L2L3)'],
            ['2D (L1)', '3D (L1L2L3)'],
            ['2D (L1)', '3D (L1L2L3)'],
            ['2D (L1)', '3D (L1L2L3)'],
            ['Lowpass', 'Broadband']]

xticks_list = [['100', '20', '5', '1'],
            ['100', '20', '5', '1'],
            ['100', '20', '5', '1'],
            ['100', '20', '5', '1'],
            ['SP', 'QP', 'L1', 'L1L2L3']]

xlabel_list = [ r'$\Delta t$' + ' in ms',
                r'$\Delta t$' + ' in ms',
                r'$\Delta t$' + ' in ms',
                r'$\Delta t$' + ' in ms',
                'Active loudspeaker set']

title_list = ['Pink noise: ' + r'$L = 250$' + ' ms',
            'Pink noise: ' + r'$L = 0.5$' + ' ms',
            'Vocal: ' + r'$L = 250$' + ' ms',
            'Vocal: ' + r'$L = 0.5$' + ' ms',
            'Pink noise: ' + r'$\Delta t = 1$' + ' ms,  '  + r'$L = 250$' + ' ms']

# Subplots of trials 0,1,2,3
fig, axs = plt.subplots(1,4, figsize=(16, 4), sharey=True)
markersz = mkr_size
fs_title = fs_labels
linest=':'
xaxis = np.arange(0,4)

order = [1,0,3,2]
for idx in range(4):
    tr = order[idx]
    axs[idx].plot(xaxis-0.1,median_all_ratings[:4,tr], label=label_list[tr][0], marker='o', markersize=markersz, linestyle=lev_linest, color='k',  markerfacecolor=mkr_color_2d, markeredgecolor='k', zorder=3)
    axs[idx].errorbar(xaxis-0.1,y=median_all_ratings[:4,tr] , capsize=4.0 ,linestyle='none', xerr=0, yerr=asymmetric_iqr[:,:4,tr] ,color='k', zorder=2)

    axs[idx].plot(xaxis+0.1,median_all_ratings[4:,tr], label=label_list[tr][1], marker='o', markersize=markersz, linestyle=linest, color='k', markerfacecolor=mkr_color_3d, markeredgecolor='k', zorder=3)
    axs[idx].errorbar(xaxis+0.1,y=median_all_ratings[4:,tr] , capsize=4.0 ,linestyle='none', xerr=0, yerr=asymmetric_iqr[:,4:,tr] ,color='k', zorder=2)

    axs[idx].set_ylim(-5,105)
    axs[idx].set_xlim(-0.5,3.5)
    axs[idx].set_xticks([0,1,2,3], xticks_list[tr], fontsize=fs_labels)
    axs[idx].set_xlabel(xlabel_list[tr], fontsize=fs_labels)
    axs[idx].set_yticks([0,25,50,75,100], ['0', '25', '50', '75', '100'], fontsize=fs_labels)
    if idx==0:
        axs[idx].set_ylabel('Envelopment', fontsize=fs_labels)
        axs[idx].legend(framealpha=1, loc='lower right', fontsize=fs_labels)
    axs[idx].set_title(title_list[tr], fontsize=fs_labels)
    axs[idx].grid(alpha=0.5)

plt.tight_layout()
plt.savefig(fname=pjoin(figures_dir,'envelopment_temporal_density.pdf'), format='pdf', bbox_inches='tight')
plt.show(block=BLOCK_PLT_SHOW)


# trial 5
plt.figure(figsize=(5, 4))
markersz = 12
fs_labels = 14
fs_title = 12
linest=':'
xaxis = np.arange(0,4)

order = [5]
for idx in range(1):
    tr = order[idx]
    plt.plot(xaxis-0.1,median_all_ratings[:4,tr], label=label_list[tr-1][0], marker='o', markersize=markersz, linestyle=lowpass_linest, color='k', markerfacecolor=lowpass_color, markeredgecolor='k', zorder=3)
    plt.errorbar(xaxis-0.1,y=median_all_ratings[:4,tr] , capsize=4.0 ,linestyle='none', xerr=0, yerr=asymmetric_iqr[:,:4,tr] ,color='k', zorder=2)

    plt.plot(xaxis+0.1,median_all_ratings[4:,tr], label=label_list[tr-1][1], marker='o', markersize=markersz, linestyle=lev_linest, color='k', markerfacecolor=broadband_color, markeredgecolor='k', zorder=3)
    plt.errorbar(xaxis+0.1,y=median_all_ratings[4:,tr] , capsize=4.0 ,linestyle='none', xerr=0, yerr=asymmetric_iqr[:,4:,tr] ,color='k', zorder=2)

    tr = tr - 1
    plt.ylim(-5,105)
    plt.xlim(-0.5,3.5)
    plt.xticks([0,1,2,3], xticks_list[tr], fontsize=fs_labels)
    plt.xlabel(xlabel_list[tr], fontsize=fs_labels)
    plt.yticks([0,25,50,75,100], ['0', '25', '50', '75', '100'], fontsize=fs_labels)
    if idx==0:
        plt.ylabel('Envelopment', fontsize=fs_labels)
    plt.title(title_list[tr], fontsize=fs_labels)
    plt.grid(alpha=0.5)
    plt.legend(framealpha=1, loc='lower right', fontsize=fs_labels)

plt.tight_layout()
plt.savefig(fname=pjoin(figures_dir,'envelopment_directional_density.pdf'), format='pdf', bbox_inches='tight')
plt.show(block=BLOCK_PLT_SHOW)



