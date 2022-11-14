import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from os.path import dirname, join as pjoin
from Utility.interauralCoherence import compute_IC_Welch, compute_IC

format = '.eps'

# FIX LENGTHs of EVAL STIMULI, TempDensity only 2 sec. long
#num_blocks = 22

root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'BinauralEvaluationAudio')
utility_dir = pjoin(root_dir, 'Utility')
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

title_list = [
    r'$L = 0.5$' + ' ms,  ' + r'$Q = 5$' + ' sec.',
    r'$\Delta t = 1$' + ' ms,  ' + r'$Q = 5$' + ' sec.',
    r'$\Delta t = 1$' + ' ms,  ' + r'$L = 250$' + ' ms',
    r'$\Delta t = 5$' + ' ms,  ' + r'$L = 250$' + ' ms'
]

IC = np.load(pjoin(utility_dir, 'IC.npy'))
PowerSpectrum = np.load(pjoin(utility_dir, 'PowerSpectrum.npy'))

scale = 2.5
fig, axs = plt.subplots(ncols=4,
                        nrows=2,
                        figsize=(6 * scale, 2 * scale),
                        sharex=True,
                        gridspec_kw={
                            'wspace': 0.15,
                            'hspace': 0.15
                        })
for eval_idx in range(4):
    EVAL = eval_list[eval_idx]

    if EVAL == 'EVAL_MaxGrainDelay':
        filelist = [
            name +
            '_Uniform_2D_MaxGrainDelay_5ms_DeltaT_1ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
            name +
            '_Uniform_2D_MaxGrainDelay_50ms_DeltaT_1ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
            name +
            '_Uniform_2D_MaxGrainDelay_500ms_DeltaT_1ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
            name +
            '_Uniform_2D_MaxGrainDelay_5000ms_DeltaT_1ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
            'DiffuseFieldReference_BINAURAL.wav'
        ]
    if EVAL == 'EVAL_GrainLength':
        filelist = [
            name +
            '_Uniform_2D_MaxGrainDelay_5000ms_DeltaT_1ms_JitterPercent_0_GrainLength_0ms_BINAURAL.wav',
            name +
            '_Uniform_2D_MaxGrainDelay_5000ms_DeltaT_1ms_JitterPercent_0_GrainLength_2ms_BINAURAL.wav',
            name +
            '_Uniform_2D_MaxGrainDelay_5000ms_DeltaT_1ms_JitterPercent_0_GrainLength_10ms_BINAURAL.wav',
            name +
            '_Uniform_2D_MaxGrainDelay_5000ms_DeltaT_1ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
            'DiffuseFieldReference_BINAURAL.wav'
        ]
    if EVAL == 'EVAL_Layers':
        filelist = [
            name +
            '_L1_MaxGrainDelay_5000ms_DeltaT_5ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
            name +
            '_L2_MaxGrainDelay_5000ms_DeltaT_5ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
            name +
            '_L3_MaxGrainDelay_5000ms_DeltaT_5ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
            name +
            '_ZEN_MaxGrainDelay_5000ms_DeltaT_5ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
            'DiffuseFieldReference_BINAURAL.wav'
        ]

    if EVAL == 'EVAL_TempDensity':
        filelist = [
            name +
            '_Uniform_2D_MaxGrainDelay_5000ms_DeltaT_1ms_JitterPercent_0_GrainLength_0ms_BINAURAL.wav',
            name +
            '_Uniform_2D_MaxGrainDelay_5000ms_DeltaT_5ms_JitterPercent_0_GrainLength_0ms_BINAURAL.wav',
            name +
            '_Uniform_2D_MaxGrainDelay_5000ms_DeltaT_20ms_JitterPercent_0_GrainLength_0ms_BINAURAL.wav',
            name +
            '_Uniform_2D_MaxGrainDelay_5000ms_DeltaT_100ms_JitterPercent_0_GrainLength_0ms_BINAURAL.wav',
            'DiffuseFieldReference_BINAURAL.wav'
        ]

    #filelist.reverse()
    num_stimuli = len(filelist)

    if EVAL == 'EVAL_GrainLength':
        labels = ['0.5 ms', '2 ms', '10 ms', '250 ms', 'ref.']
    if EVAL == 'EVAL_MaxGrainDelay':
        labels = ['5 ms', '50 ms', '500 ms', '5 sec.', 'ref.']
    if EVAL == 'EVAL_Layers':
        labels = ['L1', 'L2', 'L3', 'ZEN', 'ref.']
    if EVAL == 'EVAL_TempDensity':
        labels = ['1 ms', '5 ms', '20 ms', '100 ms', 'ref.']

    dB_offsets = [3, 2, 1, 0, -1]
    cmap = matplotlib.cm.get_cmap('cool')
    colors = [cmap(0.2), cmap(0.4), cmap(0.6), cmap(0.8), 'k']
    linestyles = ['-', '--', '-', ':', '-']

    if EVAL == 'EVAL_Layers':
        colors = [cmap(0.2), cmap(0.6), cmap(0.8), 'k', 'k']
        linestyles = ['-', '--', '-', ':', '-']
        dB_offsets = [3, 2, 1, 0, -1]

    fs = 48000
    f = np.linspace(0, fs / 2, int(blocksize / 2 + 1))
    if EVAL == 'EVAL_TempDensity':
        axs[0, eval_idx].plot([], [], ' ', label=r'$\Delta t =$')
        axs[1, eval_idx].plot([], [], ' ', label=r'$\Delta t =$')
    if EVAL == 'EVAL_GrainLength':
        axs[0, eval_idx].plot([], [], ' ', label=r'$L =$')
        axs[1, eval_idx].plot([], [], ' ', label=r'$L =$')
    if EVAL == 'EVAL_MaxGrainDelay':
        axs[0, eval_idx].plot([], [], ' ', label=r'$Q =$')
        axs[1, eval_idx].plot([], [], ' ', label=r'$Q =$')
    if EVAL == 'EVAL_Layers':
        axs[0, eval_idx].plot([], [], ' ', label='Layer:')
        axs[1, eval_idx].plot([], [], ' ', label='Layer:')

    for idx in range(num_stimuli):
        # Plot interaural coherence of stimulus / 2D diffuse field reference
        axs[0, eval_idx].semilogx(f_c,
                                  IC[eval_idx, idx, :],
                                  label=labels[idx],
                                  color=colors[idx],
                                  ls=linestyles[idx],
                                  linewidth=1.5)
        # Plot spectrum difference between stimulus and 2D diffuse field reference
        axs[1, eval_idx].semilogx(
            f_c,
            10 * np.log10(np.abs(PowerSpectrum[eval_idx, idx, :])) -
            10 * np.log10(np.abs(PowerSpectrum[eval_idx, 4, :])),
            label=labels[idx],
            color=colors[idx],
            ls=linestyles[idx],
            linewidth=1.5)

    axs[0, eval_idx].set_xlim(50, 20000)
    axs[0, eval_idx].set_ylim(0, 1.1)
    axs[0, eval_idx].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    #axs[0, eval_idx].set_ylim(-0.1, 1)
    axs[0, 0].set_ylabel('Interaural Coherence')
    #axs[0, 0].set_ylabel('IC Diff.')
    axs[0, eval_idx].set_title(title_list[eval_idx])

    axs[1, eval_idx].set_xlim(50, 20000)
    if EVAL == 'EVAL_Layers':
        axs[1, eval_idx].set_ylim(-6, 12)
        axs[1, eval_idx].set_yticks([-6, -3, 0, 3, 6, 9, 12])
    else:
        axs[1, eval_idx].set_ylim(-9, 9)
        axs[1, eval_idx].set_yticks([-9, -6, -3, 0, 3, 6, 9])
    axs[1, 0].set_ylabel('Spectral Diff. in dB')
    axs[1, eval_idx].set_xlabel('Frequency in Hz')

    axs[0, eval_idx].grid()
    axs[1, eval_idx].grid()
    if EVAL == 'EVAL_Layers':
        axs[0, eval_idx].legend(framealpha=1.0,
                                loc='upper left',
                                handlelength=1.0)
    elif EVAL == 'EVAL_TempDensity':
        axs[0, eval_idx].legend(framealpha=1.0,
                                loc='upper right',
                                handlelength=1.0)
    else:
        axs[0, eval_idx].legend(framealpha=1.0,
                                loc='upper right',
                                handlelength=1.0)

#plt.show(block=True)

#ncol=6,
#bbox_to_anchor=(0.5, 1.05),
#columnspacing=0.4,
#handletextpad=0.2,
#borderpad=0.1,
plt.savefig(fname=pjoin(save_dir, 'GranularEvaluationPlot' + format),
            bbox_inches='tight')
print('done')
