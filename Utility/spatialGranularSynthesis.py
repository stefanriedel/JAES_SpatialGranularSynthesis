import numpy as np

def hann_window(num_samples):
    return np.sin(np.linspace(0,1,num_samples) * np.pi)**2    


def spatialGranularSynthesis(x, delta_t, grain_length, Q, num_channels, output_length, fs, jitter=0, offset=0, gain=1):
    """ Multichannel granular synthesis function.
    Args:
        x (ndarray): single-channel audio input buffer
        delta_t (float): time between grains in sec.
        grain_length (float): length of grains in sec.
        Q (float): seed range (maximum grain delay) in sec.
        num_channels (int): number of output channels
        output_length (float): length of the synthesis output in sec.
        fs (float): sample rate in Hz
        jitter (float): relative jitter amount in percent, default = 0.0
        offset (int): sample read offset, default = 0
        gain (float): output gain, default = 1.0

    Returns:
        ndarray: y [int(output_length*fs), num_channels]
    """

    N = int(output_length*fs)
    y = np.zeros((N,num_channels), dtype=float)
    seed_range = int(Q*fs)

    grain_samples = int(grain_length*fs)
    window = hann_window(grain_samples)                                                     
    window_gain =  np.sum(window) / grain_samples
    delta_samples = int(delta_t*fs)

    assert(seed_range < (x.size - grain_samples))

    t = 0
    while (t+grain_samples < N):
        read_idx = np.random.randint(0, seed_range)
        grain = np.copy(x[(offset+read_idx):(offset+read_idx+grain_samples)])
        grain = grain * window

        rand_dir_idx = np.random.randint(0,num_channels)
        y[t:(t+grain_samples), rand_dir_idx] += grain

        delta = delta_samples + int(2*(np.random.rand(1)-0.5)*delta_samples*jitter)
        t += delta
    norm_factor = np.sqrt(grain_length/delta_t)*np.sqrt(window_gain)
    y = y / norm_factor
    y = y * gain

    return y