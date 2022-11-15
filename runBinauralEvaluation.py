import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from os.path import dirname, join as pjoin
from Utility.interauralCoherence import compute_IC_Welch, compute_IC
from Utility.eval_lists import eval_file_lists

format = '.eps'

root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'BinauralEvaluationAudio')
utility_dir = pjoin(root_dir, 'Utility')
save_dir = pjoin(utility_dir, 'EVAL_NPY')

# Load gammatone magnitude windows, precomputed using the 'pyfilterbank' library
# https://github.com/SiggiGue/pyfilterbank
filename = 'gammatone_erb_mag_windows_nfft_4096_numbands_320.npy'
gammatone_mag_win = np.load(pjoin(utility_dir, filename))
Nfft = int((gammatone_mag_win.shape[1] - 1) * 2)
num_bands = gammatone_mag_win.shape[0]
filename = 'gammatone_fc_numbands_320_fs_48000.npy'
f_c = np.load(pjoin(utility_dir, filename))
blocksize = 4096

# Sound signal type, pink noise in our evaluations
name = 'Pink'

eval_list = [
    'EVAL_TempDensity',
    'EVAL_GrainLength',
    'EVAL_MaxGrainDelay',
    'EVAL_Layers',
]

num_evaluations = len(eval_list)

for eval_idx in range(num_evaluations):
    EVAL = eval_list[eval_idx]

    filelist = eval_file_lists[EVAL]
    num_stimuli = len(filelist)

    IC = np.zeros((num_stimuli, num_bands))
    ILD = np.zeros((num_stimuli, num_bands))
    P_L = np.zeros((num_stimuli, num_bands))

    overlap = 1
    hopsize = int(blocksize / overlap)

    filename = name + filelist[0]
    fs, x = wavfile.read(pjoin(data_dir, filename))
    x = x / np.max(np.abs(x))
    if (x[:, 0].shape[0] != x[:, 1].shape[0]):
        assert ('left and right signals should be equal length')
    num_blocks = int(np.floor(x.shape[0] / hopsize)) - 1

    for st in range(num_stimuli):
        filename = name + filelist[st]
        fs, x = wavfile.read(pjoin(data_dir, filename))

        x_L = x[:, 0]
        x_R = x[:, 1]

        IC[st, :], ILD[st, :], P_L[st, :] = compute_IC(x_L, x_R,
                                                       gammatone_mag_win, fs,
                                                       blocksize, hopsize,
                                                       num_blocks)

        # normalization at 2 kHz frequency band
        P_L[st, :] /= P_L[st, np.where(f_c >= 2000)[0][0]]

    np.save(pjoin(save_dir, 'IC_' + EVAL + '.npy'), arr=IC)
    np.save(pjoin(save_dir, 'ILD_' + EVAL + '.npy'), arr=ILD)
    np.save(pjoin(save_dir, 'P_L_' + EVAL + '.npy'), arr=P_L)

# Compute evaluation on diffuse field reference
EVAL = 'DiffuseField'

overlap = 1
hopsize = int(blocksize / overlap)

filename = 'DiffuseFieldReference_BINAURAL.wav'
fs, x = wavfile.read(pjoin(data_dir, filename))
x = x / np.max(np.abs(x))
if (x[:, 0].shape[0] != x[:, 1].shape[0]):
    assert ('left and right signals should be equal length')
num_blocks = int(np.floor(x.shape[0] / hopsize)) - 1

fs, x = wavfile.read(pjoin(data_dir, filename))

x_L = x[:, 0]
x_R = x[:, 1]

IC, ILD, P_L = compute_IC(x_L, x_R, gammatone_mag_win, fs, blocksize, hopsize,
                          num_blocks)

# normalization at 2 kHz frequency band
P_L /= P_L[np.where(f_c >= 2000)[0][0]]

np.save(pjoin(save_dir, 'IC_' + EVAL + '.npy'), arr=IC)
np.save(pjoin(save_dir, 'ILD_' + EVAL + '.npy'), arr=ILD)
np.save(pjoin(save_dir, 'P_L_' + EVAL + '.npy'), arr=P_L)

print('done')
