import numpy as np

def compute_IC_Welch(x_L, x_R, gammatone_mag_win, fs, blocksize, hopsize, num_blocks):
    """ Compute interaural coherence (IC) and monaural ear spectrum using Bartlett/Welch's method.
    Args:
        x_L (ndarray): left-ear input signal in time-domain
        x_R (ndarray): right-ear input signal in time-domain
        gammatone_mag_win (ndarray): precomputed zero-phase magnitude bandpass windows
        fs (float): sampling rate in Hz
        blocksize (int): blocksize in samples for computation of FFT spectrum
        hopsize (int): hopsize in samples for computation of FFT spectrum
        num_blocks (int): number of blocks 

    Returns:
        ndarray: IC [num_bands]
        ndarray: P_L [num_bands]
    """
    eps=1e-6
    num_bands = gammatone_mag_win.shape[0]
    tau_limit = 0.001
    tau_range = np.arange(int(-fs*tau_limit), int(fs*tau_limit))


    cross_spectrum = np.zeros(int(blocksize/2+1), dtype=complex)
    auto_spectrum_L = np.zeros(int(blocksize/2+1), dtype=complex)
    auto_spectrum_R = np.zeros(int(blocksize/2+1), dtype=complex)

    for t in range(num_blocks):
        x_L_block = x_L[t*hopsize:(t*hopsize+blocksize)]
        x_R_block = x_R[t*hopsize:(t*hopsize+blocksize)]

        X_L = np.fft.rfft(x_L_block)
        X_R = np.fft.rfft(x_R_block)

        cross_spectrum += np.conj(X_L) * X_R
        auto_spectrum_L += np.conj(X_L) * X_L
        auto_spectrum_R += np.conj(X_R) * X_R

    cross_spectrum /= num_blocks
    auto_spectrum_L /= num_blocks
    auto_spectrum_R /= num_blocks

    IC = np.zeros(num_bands)
    ILD = np.zeros(num_bands)
    P_L = np.zeros(num_bands)
    for b in range(num_bands):
        window = gammatone_mag_win[b,:]**2
        cross_spec_w = cross_spectrum*window
        auto_spec_l_w = auto_spectrum_L*window
        auto_spec_r_w = auto_spectrum_R*window
            
        cross_correlation = np.real(np.fft.irfft(cross_spec_w))
        P_l = np.fft.irfft(auto_spec_l_w)[0]
        P_r = np.fft.irfft(auto_spec_r_w)[0]

        IC[b] = (np.max(np.abs(cross_correlation[tau_range])) + eps)  / np.sqrt((P_l + eps) * (P_r + eps))     
        ILD[b] = 10*np.log10(P_l / P_r)
        P_L[b] = P_l

    return IC, P_L