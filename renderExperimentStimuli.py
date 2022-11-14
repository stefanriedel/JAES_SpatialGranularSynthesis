import numpy as np
from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.signal as signal
from Utility.noise_psd import pink_noise
from Utility.getSubsets import getHRIR_ChannelSubset, getLoudspeaker_ChannelSubset
from spatialGranularSynthesis import spatialGranularSynthesis
from joblib import Parallel, delayed
import soundfile
import itertools
from Utility.DEQ import compute_deq, apply_deq

root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'Files')
save_dir = pjoin(root_dir, 'ExperimentStimuliAudio')
utility_dir = pjoin(root_dir, 'Utility')

# Create 2D / 3D granular synthesis stimuli used in the experiments
output_gain = 10**(-20 / 20)
fs = int(48000.0)
stimuli_length = 2.0  # in seconds
N = int(stimuli_length * fs)

# Select the sound you want to use
sound = 'Pink'
# sound = 'Vocal', requires first executing the downloadStimuliFiles.py script!
# sound = 'ConcretPH', requires first executing the downloadStimuliFiles.py script!

if sound == 'Pink':
    #Audio buffer: Pink Noise Generation
    audio_buffer = pink_noise(fs * 10)
    fc_hp = 50
    b, a = signal.butter(N=2, Wn=fc_hp, btype='hp', fs=fs, output='ba')
    audio_buffer = signal.filtfilt(b, a, audio_buffer, axis=0)
    output_name = 'Pink'

# Select the type of output you want to save to your disk:
RENDER_MULTICHANNEL_STIMULI = False
RENDER_BINAURAL_BRIR_STIMULI = False
RENDER_BINAURAL_ANECHOEIC_STIMULI = True

EXP1_STIMULI = True
EXP2_STIMULI = False

if EXP1_STIMULI:
    seed_range = [5]
    grain_lengths = [0.0005, 0.250]
    temporal_densities = [0.001, 0.005, 0.020, 0.100]
    subset = ['L1', 'L1L2L3']
if EXP2_STIMULI:
    seed_range = [5]
    grain_lengths = [0.250]
    temporal_densities = [0.020]
    subset = ['L1', 'L1L2', 'L2', 'L2L3', 'L3', 'L1L2L3', 'SP']

parameters = list(
    itertools.product(seed_range, grain_lengths, temporal_densities, subset))
num_stimuli = len(parameters)
"""[seed_range, grain_lengths, temporal_densities, subset] = parameter_lists
parameter_lists = [grain_lengths, temporal_densities, subset, seed_range]
num_stimuli = len(parameter_lists[0])
for l in parameter_lists:
    if num_stimuli != len(l):
        raise ValueError('not all parameter lists have same length!')"""

if RENDER_BINAURAL_ANECHOEIC_STIMULI:
    # Load the 2D HRIR set of the KU100 dummy head
    hrir_2D_dataset = np.load(file='./Utility/HRIR_CIRC360_48kHz.npy',
                              allow_pickle=True)
    hrir_2D = hrir_2D_dataset[0]
    hrir_l_2D = hrir_2D[:, 0, :]
    hrir_r_2D = hrir_2D[:, 1, :]

    DEQ = False
    if DEQ:
        deq_ir_l, deq_ir_r = compute_deq(hrir_l_2D, hrir_r_2D, 4096, 48000)

    # Load the 3D HRIR set of the KU100 dummy head
    hrir_3D_dataset = np.load(file='./Utility/HRIR_FULL2DEG_48kHz.npy',
                              allow_pickle=True)
    hrir_3D = hrir_3D_dataset[0]
    hrir_l_3D = hrir_3D[:, 0, :]
    hrir_r_3D = hrir_3D[:, 1, :]

    if DEQ:
        hrir_l_3D, hrir_r_3D = apply_deq(hrir_l_3D, hrir_r_3D, deq_ir_l,
                                         deq_ir_r, 4096, 48000)
        hrir_3D = np.stack((hrir_l_3D, hrir_r_3D), axis=1)


def renderBinaural(Y, hrir_l, hrir_r, gain):
    # Binaural rendering of multichannel stimulus Y

    # Assure number of channels matches
    assert (Y.shape[1] == hrir_l.shape[0])

    y_L = np.zeros(int(Y.shape[0] + hrir_l.shape[1] - 1))
    y_R = np.zeros(int(Y.shape[0] + hrir_l.shape[1] - 1))

    for idx in range(0, Y.shape[1]):
        y_L += signal.oaconvolve(Y[:, idx], hrir_l[idx, :], mode='full')
        y_R += signal.oaconvolve(Y[:, idx], hrir_r[idx, :], mode='full')
    y_L *= gain
    y_R *= gain

    y_binaural = np.array([y_L, y_R]).transpose()

    return y_binaural


def mainLoopRendering(idx):
    maximum_grain_delay = parameters[idx][0]  #seed_range[idx]
    grain_length = parameters[idx][1]  #grain_lengths[idx]
    temporal_density = parameters[idx][2]  #temporal_densities[idx]
    angular_distribution = parameters[idx][3]  #subset[idx]

    jitter = 0.01  # 1 percent temporal jitter used in the experiment.
    azi_ele, num_channels = getLoudspeaker_ChannelSubset(angular_distribution)
    Y = spatialGranularSynthesis(audio_buffer, temporal_density, grain_length,
                                 maximum_grain_delay, num_channels,
                                 stimuli_length, fs, jitter)

    if RENDER_MULTICHANNEL_STIMULI:
        output_filename = output_name + '_' + angular_distribution + '_MaxGrainDelay_' + str(
            int(maximum_grain_delay * 1000)) + 'ms_DeltaT_' + str(
                int(temporal_density * 1000)) + 'ms_JitterPercent_' + str(
                    int(jitter * 100.0)) + '_GrainLength_' + str(
                        int(grain_length * 1000)) + 'ms_MULTICHANNEL.wav'
        output_path = pjoin(save_dir, output_filename)
        soundfile.write(output_path, Y, fs)

    if RENDER_BINAURAL_ANECHOEIC_STIMULI:
        hrir_l, hrir_r, num_channels = getHRIR_ChannelSubset(
            angular_distribution, hrir_2D, hrir_3D)
        y_binaural = renderBinaural(Y, hrir_l, hrir_r, output_gain)
        if DEQ:
            binaural_string = 'BINAURAL_ANECHOEIC_DEQ'
        else:
            binaural_string = 'BINAURAL_ANECHOEIC'

        output_filename = output_name + '_' + angular_distribution + '_MaxGrainDelay_' + str(
            int(maximum_grain_delay * 1000)) + 'ms_DeltaT_' + str(
                int(temporal_density * 1000)) + 'ms_JitterPercent_' + str(
                    int(jitter * 100.0)) + '_GrainLength_' + str(
                        int(grain_length *
                            1000)) + 'ms_' + binaural_string + '.wav'
        output_path = pjoin(save_dir, output_filename)
        soundfile.write(output_path, y_binaural, fs)

    if RENDER_BINAURAL_BRIR_STIMULI:
        hrir_l, hrir_r, num_channels = getHRIR_ChannelSubset(
            angular_distribution, hrir_2D, hrir_3D)
        y_binaural = renderBinaural(Y, hrir_l, hrir_r, output_gain)
        output_filename = output_name + '_' + angular_distribution + '_MaxGrainDelay_' + str(
            int(maximum_grain_delay * 1000)) + 'ms_DeltaT_' + str(
                int(temporal_density * 1000)) + 'ms_JitterPercent_' + str(
                    int(jitter * 100.0)) + '_GrainLength_' + str(
                        int(grain_length * 1000)) + 'ms_BINAURAL_BRIR.wav'
        output_path = pjoin(save_dir, output_filename)
        soundfile.write(output_path, y_binaural, fs)

    print('File being saved to .\ExperimentStimuliAudio now. \n')

    return


print('Main rendering loop started... \n')
Parallel(n_jobs=4)(delayed(mainLoopRendering)(idx)
                   for idx in range(num_stimuli))
print('All jobs done.')
