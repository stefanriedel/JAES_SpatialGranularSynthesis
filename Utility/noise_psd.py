import numpy as np


def noise_psd(N, random, psd=lambda f: 1):
    #X_white = np.fft.rfft(np.random.randn(N));
    X_white = np.fft.rfft(2 * (random.random(N) - 0.5))
    S = psd(np.fft.rfftfreq(N))
    S = S / np.sqrt(np.mean(S**2))
    X_shaped = X_white * S
    return np.fft.irfft(X_shaped)


def PSDGenerator(f):
    return lambda N, random: noise_psd(N, random, f)


@PSDGenerator
def white_noise(f):
    return 1


@PSDGenerator
def blue_noise(f):
    return np.sqrt(f)


@PSDGenerator
def violet_noise(f):
    return f


@PSDGenerator
def brownian_noise(f):
    return 1 / np.where(f == 0, float('inf'), f)


@PSDGenerator
def pink_noise(f):
    return 1 / np.where(f == 0, float('inf'), np.sqrt(f))
