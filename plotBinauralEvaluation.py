import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from os.path import dirname, join as pjoin

format = '.pdf'

# FIX LENGTHs of EVAL STIMULI, TempDensity only 2 sec. long
#num_blocks = 22

root_dir = dirname(__file__)
utility_dir = pjoin(root_dir, 'Utility')
data_dir = pjoin(utility_dir, 'EVAL_NPY')
save_dir = pjoin(root_dir, 'Figures', 'BinauralEvaluation')

# Load gammatone magnitude windows, precomputed using the 'pyfilterbank' library
# https://github.com/SiggiGue/pyfilterbank
filename = 'gammatone_erb_mag_windows_nfft_4096_numbands_320.npy'
gammatone_mag_win = np.load(pjoin(utility_dir, filename))
Nfft = int((gammatone_mag_win.shape[1] - 1) * 2)
num_bands = gammatone_mag_win.shape[0]
filename = 'gammatone_fc_numbands_320_fs_48000.npy'
f_c = np.load(pjoin(utility_dir, filename))
blocksize = 4096

name = 'Pink'

eval_list = [
    'EVAL_TempDensity',
    'EVAL_GrainLength',
    'EVAL_MaxGrainDelay',
    'EVAL_Layers',
]

title_list = {
    'EVAL_TempDensity': r'$L = 250$' + ' ms,  ' + r'$Q = 5$' + ' sec.',
    'EVAL_GrainLength': r'$\Delta t = 1$' + ' ms,  ' + r'$Q = 5$' + ' sec.',
    'EVAL_MaxGrainDelay': r'$\Delta t = 1$' + ' ms,  ' + r'$L = 250$' + ' ms',
    'EVAL_Layers': r'$\Delta t = 5$' + ' ms,  ' + r'$L = 250$' + ' ms'
}

IC_DiffuseField = np.load(pjoin(data_dir, 'IC_DiffuseField.npy'))
ITD_DiffuseField = np.load(pjoin(data_dir, 'ITD_DiffuseField.npy'))
ILD_DiffuseField = np.load(pjoin(data_dir, 'ILD_DiffuseField.npy'))
P_L_DiffuseField = np.load(pjoin(data_dir, 'P_L_DiffuseField.npy'))

scale = 2.5
gs_kw = dict(width_ratios=[1, 1, 1, 1],
             height_ratios=[1, 0.5, 0.5, 1],
             wspace=0.2,
             hspace=0.2)
fig, axs = plt.subplots(
    ncols=4,
    nrows=4,  #nrows=2,
    figsize=(6 * scale, 2.5 * scale),  #figsize=(6 * scale, 2 * scale),
    sharex=True,
    gridspec_kw=gs_kw)
for eval_idx in range(4):
    EVAL = eval_list[eval_idx]

    IC = np.load(pjoin(data_dir, 'IC_' + EVAL + '.npy'))
    ITD = np.load(pjoin(data_dir, 'ITD_' + EVAL + '.npy'))
    ILD = np.load(pjoin(data_dir, 'ILD_' + EVAL + '.npy'))
    P_L = np.load(pjoin(data_dir, 'P_L_' + EVAL + '.npy'))

    num_stimuli = IC.shape[0]

    dB_offsets = [3, 2, 1, 0, -1]
    cmap = matplotlib.cm.get_cmap('cool')
    colors = [cmap(0.2), cmap(0.4), cmap(0.6), cmap(0.8), 'k']
    linestyles = ['-', '--', '-', ':', '-']

    if EVAL == 'EVAL_Layers':
        colors = [cmap(0.2), cmap(0.6), cmap(0.8), cmap(0.4), 'k', 'k']
        linestyles = ['-', '--', '-', '-', ':', '-']
        dB_offsets = [3, 2, 1, 0, -1]

    fs = 48000
    f = np.linspace(0, fs / 2, int(blocksize / 2 + 1))

    if EVAL == 'EVAL_TempDensity':
        labels = [
            r'$\Delta t = 1$' + ' ms', r'$\Delta t = 5$' + ' ms',
            r'$\Delta t = 20$' + ' ms', r'$\Delta t = 100$' + ' ms', 'ref.'
        ]
    if EVAL == 'EVAL_GrainLength':
        labels = [
            r'$L = 0.5$' + ' ms', r'$L = 2$' + ' ms', r'$L = 10$' + ' ms',
            r'$L = 250$' + ' ms', 'ref.'
        ]
    if EVAL == 'EVAL_MaxGrainDelay':
        labels = [
            r'$Q = 5$' + ' ms', r'$Q = 50$' + ' ms', r'$Q = 500$' + ' ms',
            r'$Q = 5$' + ' sec.', 'ref.'
        ]
    if EVAL == 'EVAL_Layers':
        labels = ['L1', 'L2', 'L3', 'SP', 'ZEN', 'ref.']

    for idx in range(num_stimuli):
        # Plot interaural coherence of stimulus
        axs[0, eval_idx].semilogx(f_c,
                                  IC[idx, :],
                                  label=labels[idx],
                                  color=colors[idx],
                                  ls=linestyles[idx],
                                  linewidth=1.5)
        # Plot spectrum difference between stimulus
        axs[3, eval_idx].semilogx(f_c,
                                  10 * np.log10(np.abs(P_L[idx, :])) -
                                  10 * np.log10(np.abs(P_L_DiffuseField)),
                                  label=labels[idx],
                                  color=colors[idx],
                                  ls=linestyles[idx],
                                  linewidth=1.5)
        # Plot ITD of stimulus
        axs[1, eval_idx].semilogx(f_c,
                                  ITD[idx, :],
                                  label=labels[idx],
                                  color=colors[idx],
                                  ls=linestyles[idx],
                                  linewidth=1.5)
        # Plot ILD of stimulus
        axs[2, eval_idx].semilogx(f_c,
                                  ILD[idx, :],
                                  label=labels[idx],
                                  color=colors[idx],
                                  ls=linestyles[idx],
                                  linewidth=1.5)
    # Plot interaural coherence of 2D diffuse field reference
    axs[0, eval_idx].semilogx(f_c,
                              IC_DiffuseField,
                              label=labels[-1],
                              color=colors[-1],
                              ls=linestyles[-1],
                              linewidth=1.5)
    # Plot spectrum difference (zero line)
    axs[3, eval_idx].semilogx(f_c,
                              10 * np.log10(np.abs(P_L_DiffuseField)) -
                              10 * np.log10(np.abs(P_L_DiffuseField)),
                              label=labels[-1],
                              color=colors[-1],
                              ls=linestyles[-1],
                              linewidth=1.5)
    # Plot ITD of ref
    axs[1, eval_idx].semilogx(f_c,
                              ITD_DiffuseField,
                              label=labels[-1],
                              color=colors[-1],
                              ls=linestyles[-1],
                              linewidth=1.5)
    # Plot ILD of ref
    axs[2, eval_idx].semilogx(f_c,
                              ILD_DiffuseField,
                              label=labels[-1],
                              color=colors[-1],
                              ls=linestyles[-1],
                              linewidth=1.5)

    axs[0, eval_idx].set_xlim(50, 20000)
    axs[0, eval_idx].set_ylim(0, 1.05)
    axs[0, eval_idx].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[0, 0].set_ylabel(r'$\overline{\mathrm{IC}}$', fontsize=11)
    #axs[0, 0].set_ylabel('Mean IC', fontsize=12)
    axs[0, eval_idx].set_title(title_list[EVAL])

    axs[3, eval_idx].set_xlim(50, 20000)
    axs[3, eval_idx].set_ylim(-9, 9)
    axs[3, eval_idx].set_yticks([-9, -9, -6, -3, 0, 3, 6, 9, 9])

    #axs[1, 0].set_ylabel('Mean Spectral Diff. [dB]')

    #axs[1, eval_idx].set_yticks([0, 3, 6, 9, 12])
    #axs[1, eval_idx].set_ylim(0, 12)
    axs[1, eval_idx].set_yticks([0, 0.0005, 0.001])
    axs[1, eval_idx].set_yticklabels(['0', '0.5', '1'])
    axs[1, eval_idx].set_ylim(-0.5e-4, 0.001)
    axs[1, 0].set_ylabel(r'$\mathrm{ITD}_\mathrm{SD}$' + ' [ms]', fontsize=11)

    axs[2, eval_idx].set_yticks([0, 3, 6, 9])
    axs[2, eval_idx].set_ylim(-0.5, 9)
    axs[2, 0].set_ylabel(r'$\mathrm{ILD}_\mathrm{SD}$' + ' [dB]', fontsize=11)

    axs[3, 0].set_ylabel(
        r'$\overline{\xi_\mathrm{L}} - \overline{\xi_\mathrm{L,ref}}$' +
        ' [dB]',
        fontsize=11)
    axs[3, eval_idx].set_xlabel('Frequency in Hz', fontsize=11)

    axs[0, eval_idx].grid()
    axs[1, eval_idx].grid()
    axs[2, eval_idx].grid()
    axs[3, eval_idx].grid()
    """axs[0, eval_idx].legend(framealpha=1.0,
                            loc='upper left',
                            bbox_to_anchor=(-0.02, 1.03),
                            ncol=3,
                            handlelength=1.0,
                            handletextpad=0.05,
                            columnspacing=0.9,
                            labelspacing=0.05)"""

    axs[3, eval_idx].legend(framealpha=1.0,
                            loc='upper left',
                            bbox_to_anchor=(-0.02, 1.03),
                            ncol=3,
                            handlelength=1.0,
                            handletextpad=0.05,
                            columnspacing=0.9,
                            labelspacing=0.05)

plt.savefig(fname=pjoin(save_dir, 'GranularEvaluationPlot' + format),
            bbox_inches='tight')
print('done')
