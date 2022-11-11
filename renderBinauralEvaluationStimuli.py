import numpy as np
from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.signal as signal
from Utility.noise_psd import pink_noise
from Utility.getSubsets import getHRIR_ChannelSubset
from Utility.spatialGranularSynthesis import spatialGranularSynthesis
from joblib import Parallel, delayed

root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'Files')
save_dir = pjoin(root_dir, 'BinauralEvaluationAudio')
utility_dir = pjoin(root_dir, 'Utility')

# create 2D / 3D granular synthesis stimuli for binaural evaluation
output_gain = 10**(-20/20)
fs = int(48000.0)
stimuli_length = 5.0     # in seconds                                            
N = int(stimuli_length * fs) 

#Audio buffer: Pink Noise Generation
audio_buffer = pink_noise(fs * 10)
fc_hp = 50
b,a = signal.butter(N=2, Wn=fc_hp, btype='hp', fs=fs, output='ba')
audio_buffer = signal.filtfilt(b,a,audio_buffer,axis=0)
output_name = 'Pink'

# Select parameter evaluation
EVAL_GrainLength = False
EVAL_MaxGrainDelay = False
EVAL_Layers = True

RENDER_DIFFUSE_REF = True # only enable once to render the ref. file

num_cond = 4
if EVAL_GrainLength:
    maximum_grain_delays = [5] * num_cond              
    grain_lengths = [0.250,0.010,0.002,0.0005]    
    temporal_densities = [0.001] * num_cond
    angular_distributions = ['Uniform_2D'] * num_cond
if EVAL_MaxGrainDelay:
    maximum_grain_delays = [5, 0.5, 0.050, 0.005]
    grain_lengths = [0.250]  * num_cond
    temporal_densities = [0.001] * num_cond
    angular_distributions = ['Uniform_2D'] * num_cond
if EVAL_Layers:
    maximum_grain_delays = [5] * num_cond              
    grain_lengths = [0.250]  * num_cond
    temporal_densities = [0.005] * num_cond
    angular_distributions = ['L1','L2','L3','ZEN']

parameter_lists = [grain_lengths, temporal_densities, angular_distributions, maximum_grain_delays]
num_stimuli = len(parameter_lists[0])
for l in parameter_lists:
    if num_stimuli != len(l):
        raise ValueError('not all parameter lists have same length!')


# Load the 2D HRIR set of the KU100 dummy head
hrir_2D_dataset = np.load(file='./Utility/HRIR_CIRC360_48kHz.npy', allow_pickle=True)
hrir_2D = hrir_2D_dataset[0]
hrir_l_2D = hrir_2D[:,0,:]
hrir_r_2D = hrir_2D[:,1,:]
# Load the 3D HRIR set of the KU100 dummy head
hrir_3D_dataset = np.load(file='./Utility/HRIR_FULL2DEG_48kHz.npy', allow_pickle=True)
hrir_3D = hrir_3D_dataset[0]
hrir_l_3D = hrir_3D[:,0,:]
hrir_r_3D = hrir_3D[:,1,:]

if RENDER_DIFFUSE_REF:
    num_channels = 360
    Y = np.zeros((N,num_channels), dtype=float)
    for ch in range(num_channels):
        Y[:,ch] = pink_noise(N)
        Y[:,ch] = signal.filtfilt(b,a,Y[:,ch],axis=0)
    # Render and save 2D diffuse field reference
    y_L = np.zeros(int(Y.shape[0]+hrir_l_2D.shape[1]-1))
    y_R = np.zeros(int(Y.shape[0]+hrir_l_2D.shape[1]-1))

    for idx in range(0,num_channels):
        y_L += signal.fftconvolve(Y[:,idx], hrir_l_2D[idx,:])
        y_R += signal.fftconvolve(Y[:,idx], hrir_r_2D[idx,:])
    y_L /= np.sqrt(num_channels)
    y_R /= np.sqrt(num_channels)

    y_L *= output_gain
    y_R *= output_gain

    y_binaural = np.array([y_L, y_R]).transpose()
    output_filename = 'DiffuseFieldReference' + '_BINAURAL.wav'
    output_path = pjoin(save_dir, output_filename)
    wavfile.write(output_path, fs, y_binaural)


def mainLoopSimulation(idx):
    angular_distribution = angular_distributions[idx]
    grain_length = grain_lengths[idx]
    temporal_density = temporal_densities[idx]
    maximum_grain_delay = maximum_grain_delays[idx]

    hrir_l, hrir_r, num_channels = getHRIR_ChannelSubset(angular_distribution, hrir_2D, hrir_3D)
    jitter = 0
    Y = spatialGranularSynthesis(audio_buffer, temporal_density, grain_length, maximum_grain_delay, num_channels, stimuli_length, fs, jitter)

    # Binaural rendering 
    y_L = np.zeros(int(Y.shape[0]+hrir_l.shape[1]-1))
    y_R = np.zeros(int(Y.shape[0]+hrir_l.shape[1]-1))

    for idx in range(0,num_channels):
        y_L += signal.oaconvolve(Y[:,idx], hrir_l[idx,:], mode='full')
        y_R += signal.oaconvolve(Y[:,idx], hrir_r[idx,:], mode='full')
    y_L *= output_gain
    y_R *= output_gain

    y_binaural = np.array([y_L, y_R]).transpose()

    output_filename = output_name + '_' + angular_distribution + '_MaxGrainDelay_' + str(int(maximum_grain_delay * 1000)) + 'ms_DeltaT_' + str(int(temporal_density * 1000)) + 'ms_JitterPercent_' + str(int(jitter*100.0)) + '_GrainLength_' + str(int(grain_length * 1000)) +  'ms_BINAURAL.wav'

    output_path = pjoin(save_dir, output_filename)
    wavfile.write(output_path, fs, y_binaural)

    return

print('Main binaural rendering loop started... \n')
Parallel(n_jobs=4)(delayed(mainLoopSimulation)(idx) for idx in range(num_stimuli))
print('Audio Files being saved to .\BinauralAudio now.')
