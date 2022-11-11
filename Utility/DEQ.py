import numpy as np
import matplotlib.pyplot as plt


def compute_deq(hrir_l, hrir_r, blocksize, fs):
    #Compute Diffuse Field Equalization
    hrtf_l = np.fft.rfft(hrir_l, n=blocksize, axis=-1)
    hrtf_r = np.fft.rfft(hrir_r, n=blocksize, axis=-1)

    fs = int(fs)

    diffuse_response_l = np.sqrt(
        np.sum(np.abs(hrtf_l)**2, axis=0) / hrtf_l.shape[0])
    diffuse_response_r = np.sqrt(
        np.sum(np.abs(hrtf_r)**2, axis=0) / hrtf_l.shape[0])
    """f = np.linspace(0,fs/2,num=int(blocksize/2+1))
    plt.semilogx(f, 20*np.log10(np.abs(diffuse_response_l)), label='diff L')
    plt.legend()
    plt.ylim(-60,10)
    plt.show()"""

    diffuse_field_eq_L = 1 / diffuse_response_l
    diffuse_field_eq_R = 1 / diffuse_response_r

    diffuse_eq_ir_L = np.fft.irfft(diffuse_field_eq_L, axis=-1)
    diffuse_eq_ir_R = np.fft.irfft(diffuse_field_eq_R, axis=-1)

    return diffuse_eq_ir_L, diffuse_eq_ir_R


def apply_deq(hrir_l, hrir_r, diffuse_eq_ir_L, diffuse_eq_ir_R, blocksize, fs):
    fs = int(fs)

    hrtf_l = np.fft.rfft(hrir_l, n=blocksize, axis=-1)
    hrtf_r = np.fft.rfft(hrir_r, n=blocksize, axis=-1)

    diffuse_field_eq_L = np.fft.rfft(diffuse_eq_ir_L, n=blocksize)
    diffuse_field_eq_R = np.fft.rfft(diffuse_eq_ir_R, n=blocksize)

    hrtf_l = hrtf_l * np.tile(diffuse_field_eq_L, (hrtf_l.shape[0], 1))
    hrtf_r = hrtf_r * np.tile(diffuse_field_eq_R, (hrtf_l.shape[0], 1))

    hrir_l_deq = np.fft.irfft(hrtf_l, axis=-1)
    hrir_r_deq = np.fft.irfft(hrtf_r, axis=-1)

    return hrir_l_deq, hrir_r_deq
