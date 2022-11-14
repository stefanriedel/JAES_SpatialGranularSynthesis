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

num_evaluations = len(eval_list)
num_stimuli = 5  #len(filelist)
IC = np.zeros((num_evaluations, num_stimuli, num_bands))
PowerSpectrum = np.zeros((num_evaluations, num_stimuli, num_bands))

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

    overlap = 1
    hopsize = int(blocksize / overlap)

    filename = filelist[0]
    fs, x = wavfile.read(pjoin(data_dir, filename))
    x = x / np.max(np.abs(x))
    if (x[:, 0].shape[0] != x[:, 1].shape[0]):
        assert ('left and right signals should be equal length')
    num_blocks = int(np.floor(x.shape[0] / hopsize)) - 1

    for st in range(num_stimuli):
        filename = filelist[st]
        fs, x = wavfile.read(pjoin(data_dir, filename))

        x_L = x[:, 0]
        x_R = x[:, 1]

        if 0:  #eval_st < 3:
            IC[eval_idx,
               st, :], PowerSpectrum[eval_idx, st, :] = compute_IC_Welch(
                   x_L, x_R, gammatone_mag_win, fs, blocksize, hopsize,
                   num_blocks)
        else:
            IC[eval_idx, st, :], PowerSpectrum[eval_idx, st, :] = compute_IC(
                x_L, x_R, gammatone_mag_win, fs, blocksize, hopsize,
                num_blocks)

        PowerSpectrum[eval_idx, st, :] /= PowerSpectrum[
            eval_idx, st, np.where(
                f_c >= 2000)[0][0]]  # normalization at 2 kHz frequency band

np.save(pjoin(utility_dir, 'IC.npy'), arr=IC)
np.save(pjoin(utility_dir, 'PowerSpectrum.npy'), arr=PowerSpectrum)
print('done')
