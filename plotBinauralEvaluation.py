import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from os.path import dirname, join as pjoin
from Utility.interauralCoherence import compute_IC_Welch


root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'BinauralAudio')
utility_dir = pjoin(root_dir, 'Utility')
save_dir =  pjoin(root_dir, 'Figures', 'BinauralEvaluation')

# Load gammatone magnitude windows, precomputed using the 'pyfilterbank' library
# https://github.com/SiggiGue/pyfilterbank
filename = 'gammatone_erb_mag_windows_nfft_4096_numbands_320.npy'
gammatone_mag_win = np.load(pjoin(utility_dir, filename))
Nfft = int((gammatone_mag_win.shape[1]-1) * 2)
num_bands = gammatone_mag_win.shape[0]
filename = 'gammatone_fc_numbands_320_fs_48000.npy'
f_c = np.load(pjoin(utility_dir, filename))
blocksize = 4096

name = 'Pink'
EVAL_MaxGrainDelay = True
EVAL_GrainLength = False
EVAL_Layers = False

eval_list = ['EVAL_GrainLength', 'EVAL_MaxGrainDelay', 'EVAL_Layers']

title_list = ['Pink noise: ' + r'$\Delta t = 1$' + ' ms,  '  + r'$Q = 5$' + ' sec.', 
                'Pink noise: ' + r'$\Delta t = 1$' + ' ms,  '  + r'$L = 250$' + ' ms',
                'Pink noise: ' + r'$\Delta t = 5$' + ' ms,  '  + r'$L = 250$' + ' ms']


scale = 2.5
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(6*scale,2*scale), sharex=True, gridspec_kw = {'wspace':0.15, 'hspace':0.15})
for eval_idx in range(3):
    EVAL = eval_list[eval_idx]

    if EVAL ==  'EVAL_MaxGrainDelay':
        filelist = [name + '_Uniform_2D_MaxGrainDelay_5ms_DeltaT_1ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
                    name + '_Uniform_2D_MaxGrainDelay_50ms_DeltaT_1ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
                    name + '_Uniform_2D_MaxGrainDelay_500ms_DeltaT_1ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
                    name + '_Uniform_2D_MaxGrainDelay_5000ms_DeltaT_1ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
                    'DiffuseFieldReference_BINAURAL.wav']
    if EVAL ==  'EVAL_GrainLength':
        filelist = [name + '_Uniform_2D_MaxGrainDelay_5000ms_DeltaT_1ms_JitterPercent_0_GrainLength_0ms_BINAURAL.wav',
                    name + '_Uniform_2D_MaxGrainDelay_5000ms_DeltaT_1ms_JitterPercent_0_GrainLength_2ms_BINAURAL.wav',
                    name + '_Uniform_2D_MaxGrainDelay_5000ms_DeltaT_1ms_JitterPercent_0_GrainLength_10ms_BINAURAL.wav',
                    name + '_Uniform_2D_MaxGrainDelay_5000ms_DeltaT_1ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
                    'DiffuseFieldReference_BINAURAL.wav']
    if EVAL ==  'EVAL_Layers':
        filelist = [name + '_L1_MaxGrainDelay_5000ms_DeltaT_5ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
                    name + '_L2_MaxGrainDelay_5000ms_DeltaT_5ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
                    name + '_L3_MaxGrainDelay_5000ms_DeltaT_5ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
                    name + '_ZEN_MaxGrainDelay_5000ms_DeltaT_5ms_JitterPercent_0_GrainLength_250ms_BINAURAL.wav',
                    'DiffuseFieldReference_BINAURAL.wav']


    #filelist.reverse()
    num_stimuli = len(filelist)

    overlap = 1
    hopsize = int(blocksize / overlap)

    filename = filelist[0]
    fs, x = wavfile.read(pjoin(data_dir, filename))
    x = x / np.max(np.abs(x))
    if(x[:,0].shape[0] != x[:,1].shape[0]):
        assert('left and right signals should be equal length')
    num_blocks = int(np.floor(x.shape[0] / hopsize)) - 1


    IC = np.zeros((num_stimuli, num_bands))
    PowerSpectrum = np.zeros((num_stimuli, num_bands))

    IACC_values = np.zeros((num_stimuli, num_blocks))
    MagSpectrum = np.zeros((int(blocksize/2+1), num_stimuli, num_blocks), dtype=complex)
    for idx in range(num_stimuli):
        filename = filelist[idx]
        fs, x = wavfile.read(pjoin(data_dir, filename))

        x_L = x[:,0]
        x_R = x[:,1]

        IC[idx,:], PowerSpectrum[idx,:] = compute_IC_Welch(x_L, x_R, gammatone_mag_win, fs, blocksize, hopsize, num_blocks)
        PowerSpectrum[idx,:] /= PowerSpectrum[idx, np.where(f_c >= 2000)[0][0]] # normalization at 2 kHz frequency band


    if EVAL ==  'EVAL_GrainLength':
        labels = ['L = 0.5 ms', 'L = 2 ms', 'L = 10 ms', 'L = 250 ms', 'ref.']
    if EVAL ==  'EVAL_MaxGrainDelay':
        labels = ['Q = 5 ms', 'Q = 50 ms', 'Q = 500 ms', 'Q = 5 sec.', 'ref.']
    if EVAL ==  'EVAL_Layers':
        labels = ['L1', 'L2', 'L3', 'ZEN', 'ref.']

    dB_offsets = [3,2,1,0,-1]   
    cmap = matplotlib.cm.get_cmap('cool')
    colors = [cmap(0.2), cmap(0.4), cmap(0.6), cmap(0.8), 'k']
    linestyles = ['-', '--', '-',':','-']

    if EVAL == 'EVAL_Layers':
        colors = [cmap(0.2), cmap(0.6), cmap(0.8), 'k' , 'k'] 
        linestyles = ['-', '--', '-',':','-']
        dB_offsets = [3,2,1,0,-1]

    fs = 48000
    f = np.linspace(0,fs/2, int(blocksize/2+1))
    for idx in range(num_stimuli):
        # Plot interaural coherence of stimulus / 2D diffuse field reference
        axs[0,eval_idx].semilogx(f_c, IC[idx,:], label=labels[idx], color=colors[idx], ls=linestyles[idx], linewidth=1.5)
        # Plot spectrum difference between stimulus and 2D diffuse field reference
        axs[1,eval_idx].semilogx(f_c, 10*np.log10(np.abs(PowerSpectrum[idx,:])) - 10*np.log10(np.abs(PowerSpectrum[4,:])), label=labels[idx], color=colors[idx], ls=linestyles[idx], linewidth=1.5)

    axs[0,eval_idx].set_xlim(50,20000)
    axs[0,eval_idx].set_ylim(0,1)
    axs[0,0].set_ylabel('Interaural Coherence')
    axs[0,eval_idx].set_title(title_list[eval_idx])

    axs[1,eval_idx].set_xlim(50,20000)
    if EVAL == 'EVAL_Layers':
        axs[1,eval_idx].set_ylim(-6,12)
        axs[1,eval_idx].set_yticks([-6,-3,0,3,6,9,12])
    else:
        axs[1,eval_idx].set_ylim(-9,9)
        axs[1,eval_idx].set_yticks([-9,-6,-3,0,3,6,9])  
    axs[1,0].set_ylabel('Spectral Diff. in dB')
    axs[1,eval_idx].set_xlabel('Frequency in Hz')

    axs[0,eval_idx].grid()
    axs[1,eval_idx].grid()
    if EVAL ==  'EVAL_Layers':
        axs[0,eval_idx].legend(framealpha=1.0, loc='upper left')
    else:
        axs[0,eval_idx].legend(framealpha=1.0, loc='upper right')


plt.savefig(fname=pjoin(save_dir, 'GranularEvaluationPlot.pdf'), bbox_inches='tight')
print('done')



