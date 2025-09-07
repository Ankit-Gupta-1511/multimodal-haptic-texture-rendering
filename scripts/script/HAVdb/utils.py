import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def hsv2rgb(h,s,v):
    """
    Convert HSV color space to RGB color space
    
    Args:
        h (float): The hue.
        s (float): The saturation.
        v (float): The value.
        
    Returns:
        Tuple[float, float, float]: The RGB color.
    """
    if s == 0.0: v*=255; return (v,v,v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = int(255*(v*(1.-s))), int(255*(v*(1.-s*f))), int(255*(v*(1.-s*(1.-f)))); v*=255; i%=6
    v = int(v)
    if i == 0: return (v,t,p)
    if i == 1: return (q,v,p)
    if i == 2: return (p,v,t)
    if i == 3: return (p,q,v)
    if i == 4: return (t,p,v)
    if i == 5: return (v,p,q)

def generateColors(N):
    """
    Generate N colors.
    
    Args:
        N (int): The number of colors to generate.
        
    Returns:
        List[Tuple[float, float, float]]: The list of colors.
    """
    HSV_tuples = [(x*1.0/N, 0.8, 0.8) for x in range(N)]
    RGB_tuples = map(lambda x: hsv2rgb(*x), HSV_tuples)
    #divide by 255 to get the right values
    RGB_tuples = [tuple([x/255 for x in rgb]) for rgb in RGB_tuples]
    return RGB_tuples

def constant_derivative_mask(time, signal, threshold_max=200,threshold_min=0, conv_win_s=0, conv_win_d=0):
    """
    Compute the derivative of the signal and return a mask of the values where the derivative is below the threshold.
    
    Args:
        time (np.ndarray): The time array.
        signal (np.ndarray): The signal array.
        threshold (float, optional): The threshold to use. Defaults to 200.
        convolution_window (int, optional): The convolution window to use. Defaults to 0: no convolution.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: The mask and the derivative.
    """
    if conv_win_s > 0:
        signal = np.convolve(signal, np.ones(conv_win_s)/conv_win_s, mode='same')
    der = np.diff(signal,append=2*signal[-1]-signal[-2]) / np.diff(time, append=2*time[-1]-time[-2])
    if conv_win_d > 0:
        der = np.convolve(der, np.ones(conv_win_d)/conv_win_d, mode='same')
    mask_der = (np.abs(der) < threshold_max) & (np.abs(der) > threshold_min) 
    
    return mask_der, der

def ravel_angle(dir, min_angle=-180):
    """
    Ravel the given angle to the given range.
    
    Args:
        dir (np.ndarray): The angle array.
        min_angle (int, optional): The minimum angle. Defaults to -180.
        
    Returns:
        np.ndarray: The raveled angle array.
    """
    dir = np.mod(dir, 360)
    dir[dir>180] = dir[dir>180]-360
    dir[dir<-180] = dir[dir<-180]+360
    dir = np.mod(dir-min_angle, 360) + min_angle
    return dir

def filterConv(signal, window_size):
    """
    Filter the given signal with a convolution filter.
    
    Args:
        window_size (int): The size of the filter.
        
    Returns:
        np.ndarray: The filter.
    """
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')



def interpolate(uncontinuous_time, uncontinuous_data, new_sampling_rate):
    """
    Interpolate the given data to a new sampling rate.
    
    Args:
        uncontinuous_time (np.ndarray): The time array.
        uncontinuous_data (np.ndarray): The data array.
        new_sampling_rate (int): The new sampling rate.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: The interpolated time and data arrays.
    """
    new_time = np.arange(uncontinuous_time[0], uncontinuous_time[-1], 1/new_sampling_rate)
    new_data = interp1d(uncontinuous_time, uncontinuous_data, axis=0)(new_time)
    return new_time, new_data

def plot_distances(distances, delim, ytck, Y):
    plt.figure(figsize=(15,15))
    plt.imshow(distances, aspect='auto', cmap='jet')
    for d in delim:
        plt.axhline(d+0.5, color='black', linewidth=1)
        plt.axvline(d+0.5, color='black', linewidth=1)
    plt.yticks(ytck, [Y[int(i),0] for i in ytck])
    plt.title("FFT distances")
    
def plot_ffts(ffts, delim, ytck, Y):
    plt.figure(figsize=(15,15))
    plt.imshow(ffts, aspect='auto', cmap='jet', interpolation='none')
    for d in delim:
        plt.axhline(d+0.5, color='black', linewidth=1)
    plt.yticks(ytck, [Y[int(i),0] for i in ytck])

def compute_distance(X):
    distances = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i,X.shape[0]):
            distances[i,j] = np.linalg.norm(X[i,:]-X[j,:])
            distances[j,i] = distances[i,j]
    return distances

def plotSpectrogram(time, sig, Fs, NFFT=1024, noverlap=128, cmap='jet', title='Spectrogram', color="black"):
    plt.specgram(sig, Fs=12000, NFFT=NFFT, noverlap=noverlap, cmap=cmap)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.plot(time, ((sig-np.min(sig))/(np.max(sig)-np.min(sig))+1)*Fs/4, color=color)
    plt.colorbar(label='Intensity [dB]')