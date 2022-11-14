import numpy as np
import scipy.signal as signal


def hann_window(num_samples):
    return np.sin(np.linspace(0, 1, num_samples) * np.pi)**2


def multichannelGranularSynthesis(x,
                                  delta_t,
                                  grain_length,
                                  seed_range,
                                  num_channels,
                                  output_length,
                                  fs,
                                  random,
                                  jitter=0,
                                  offset=0,
                                  gain=1):
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

    N = int(output_length * fs)
    y = np.zeros((N, num_channels), dtype=float)
    Q = int(seed_range * fs)

    L = int(grain_length * fs)
    window = hann_window(L)
    window_gain = np.sum(window**2) / L
    delta_samples = int(delta_t * fs)

    assert (Q < (x.size - L))

    t = 0
    while (t + L < N):
        read_idx = random.integers(0, Q)
        grain = np.copy(x[(offset + read_idx):(offset + read_idx + L)])
        grain = grain * window

        rand_dir_idx = random.integers(0, num_channels)
        y[t:(t + L), rand_dir_idx] += grain

        delta = delta_samples + int(
            2 * (np.random.rand(1) - 0.5) * delta_samples * jitter)
        t += delta
    norm_factor = np.sqrt(grain_length / delta_t) * np.sqrt(window_gain)
    y = y / norm_factor
    y = y * gain

    return y


def binauralGranularSynthesis(x,
                              delta_t,
                              grain_length,
                              seed_range,
                              output_length,
                              fs,
                              hrir,
                              random,
                              jitter=0,
                              offset=0,
                              gain=1):
    """ Binaural granular synthesis function.
    Args:
        x (ndarray): single-channel audio input buffer
        delta_t (float): time between grains in sec.
        grain_length (float): length of grains in sec.
        seed_range (float): seed range (maximum grain delay) in sec.
        num_channels (int): number of output channels
        output_length (float): length of the synthesis output in sec.
        fs (float): sample rate in Hz
        jitter (float): relative jitter amount in percent, default = 0.0
        offset (int): sample read offset, default = 0
        gain (float): output gain, default = 1.0

    Returns:
        ndarray: y [int(output_length*fs), num_channels]
    """

    num_channels = 2
    N = int(output_length * fs)
    y = np.zeros((N, num_channels), dtype=float)
    Q = int(seed_range * fs)
    num_directions = int(hrir.shape[0])
    hrir_length = int(hrir.shape[2])

    L = int(grain_length * fs)
    window = hann_window(L)
    window_gain = np.sum(window**2) / L
    delta_samples = int(delta_t * fs)

    assert (Q < (x.size - L))

    cL = L + hrir_length - 1
    t = 0
    while (t + cL < N):
        read_idx = random.integers(0, Q)
        grain = np.copy(x[(offset + read_idx):(offset + read_idx + L)])
        grain = grain * window

        rand_dir_idx = random.integers(0, num_directions)

        y[t:(t + cL), 0] += signal.fftconvolve(grain, hrir[rand_dir_idx, 0, :])
        y[t:(t + cL), 1] += signal.fftconvolve(grain, hrir[rand_dir_idx, 1, :])

        delta = delta_samples + int(
            2 * (np.random.rand(1) - 0.5) * delta_samples * jitter)
        t += delta
    norm_factor = np.sqrt(grain_length / delta_t) * np.sqrt(window_gain)
    y = y / norm_factor
    y = y * gain

    return y