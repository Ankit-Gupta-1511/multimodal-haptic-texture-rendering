import scipy
from scipy.interpolate import interp1d
import numpy as np


def get_seg_fft(segmentation, sig, sampling_rate):
    """
    Compute the fft of the signal on each segment of the segmentation.
    
    Args:
        segmentation (list): The segmentation of the signal.
        sig (np.ndarray): The signal.
        sampling_rate (int): The sampling rate.
        
    Returns:
        np.ndarray: The fft of the signal on each segment of the segmentation.
    """
    #compute the fft of the vibration and audio on each segment
    fft_list = []

    #list of frequencies from 0 to 6000 Hz
    frequencies = np.arange(0,sampling_rate/2,1)

    for i in range(len(segmentation)):
        index_start = segmentation[i][0]
        index_end = segmentation[i][1]
        #compute the fft
        sig_fft = np.abs(scipy.fft.rfft(sig[index_start:index_end,]))
        freq = scipy.fft.rfftfreq(sig[index_start:index_end,].shape[0], d=1/sampling_rate)
        #interpolate the fft to frequencies
        f_sig_fft = interp1d(freq, sig_fft, axis=0, bounds_error=False, fill_value=0)(frequencies)
        fft_list.append(f_sig_fft)
    return np.array(fft_list)