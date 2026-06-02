import random
import copy

import torch
import numpy as np
from scipy import signal


def compute_active_speech_level_detailed(signal, fs, target_level_dBm0=-26):
    """
    Compute the Active Speech Level (ASL) based on ITU-T P.56, with optional normalization.
    
    input: signal, input signal, Numpy array, (N, )
    input: fs, int, sampling rate, 16000
    input: target_level_dBm0, int, target active speech level, -26 dB

    output: asl_msq_db, float, active speech level (dB)
    output: signal, same shape as x
    output: scale, float, scale factor to makr input signal to be at the target_level
    """
    asl_msq_db, actfact, c0 = asl_P56(signal, fs)
    scale = 10 ** ((target_level_dBm0 - asl_msq_db) / 20.0)
    return signal * scale
    

def asl_P56(x, fs, nbits=16):
    """
    input: x, input signal, (N, )
    input: fs, int, sampling rate, 16000
    input: nbits, int, number of bits per sample, default 16
    """
    assert x.ndim == 1
    eps = np.finfo(float).eps
    # time constant of smoothing, in seconds
    T = 0.03 
    # hangover time in seconds
    H = 0.2  
    # margin in dB of the difference between threshold and ASL
    M = 15.9  
    # number of thresholds, for 16 bit, it's 15
    thres_no = nbits-1  
    # reference
    refdB = 0
    
    # hangover in samples
    I = np.floor(fs * H + 0.5) 
    # smoothing factor in envelop detection
    g = np.exp(-1 / (fs * T))  
    
    # 
    a = np.full(thres_no, -1)
    # threshold
    c = 2**np.arange(-15, thres_no-15, dtype=float)
    # hangover counter for each level threshold
    hang = np.full(thres_no, I)  

    # length of x
    x_len = x.shape[0] 

    # squared sum of samples since last reset 
    sq = np.sum(x * x)
    # sum 
    s_sum = np.sum(x)
    # maximum values
    max_abs = np.max(np.abs(x))
    max_wav = np.max(x)
    min_wav = np.min(x)
    
    
    # use a 2nd order IIR filter to detect the envelope q
    x_abs = np.abs(x)
    p = signal.lfilter([1-g, 0], [1, -g], np.squeeze(x_abs))
    # q is the envelope, obtained from moving average of abs(x) (with slight "hangover").
    q = signal.lfilter([1-g, 0], [1, -g], np.squeeze(p))  

    # count the number of active points
    for k in range(x_len):
        for j in range(thres_no):
            if q[k] >= c[j]:
                a[j] = a[j]+1
                hang[j] = 0
            elif hang[j] < I:
                a[j] = a[j]+1
                hang[j] = hang[j]+1
            else:
                break

    asl_ms_log = -100.0
    actfact = 0
    asl_msq = 0

    c0 = None
    
    if a[0] == -1:
        return asl_ms_log, actfact, c0
    else:
        a+=2
        AdB1=10*np.log10(sq/a[0]+eps)

    CdB1 = 20*np.log10(c[0]+eps)
    if (AdB1-CdB1 < M):
        return asl_ms_log, actfact, c0

    AdB = np.zeros(thres_no)
    CdB = np.zeros(thres_no)
    Delta = np.zeros(thres_no)
    
    AdB[0] = AdB1
    CdB[0] = CdB1
    Delta[0] = AdB1-CdB1

    for j in range(1, thres_no):
        AdB[j] = 10 * np.log10(sq/(a[j]+eps)+eps)
        CdB[j] = 20 * np.log10(c[j]+eps)

    for j in range(1, thres_no):
        if a[j] != 0:
            Delta[j]= AdB[j]- CdB[j]
            if Delta[j] <= M:
                # interpolate to find the actfact
                asl_ms_log, cl0 = bin_interp(AdB[j],
                    AdB[j-1], CdB[j], CdB[j-1], M, 0.5)
                asl_msq = 10**(asl_ms_log/10) # this is the mean square value NOT the rms
                actfact = (sq/x_len)/asl_msq # this is the proportion of the time that the speech is deemed "active"
                c0= 10**(cl0/20) # this is the threshold above which the speech is deemed "active".
                break
    
    return asl_ms_log - refdB, actfact, c0

# --------------------------------------------------------------------------

def bin_interp(upcount, lwcount, upthr, lwthr, Margin, tol):

    if tol < 0:
        tol = -tol

    # Check if extreme counts are not already the true active value
    iterno = 1
    if np.abs(upcount - upthr - Margin) < tol:
        asl_ms_log = upcount
        cc = upthr
        return asl_ms_log, cc

    if np.abs(lwcount - lwthr - Margin) < tol:
        asl_ms_log= lwcount
        cc= lwthr
        return asl_ms_log, cc

    # Initialize first middle for given (initial) bounds
    midcount = (upcount + lwcount) / 2.0
    midthr = (upthr + lwthr) / 2.0

    # Repeats loop until `diff' falls inside the tolerance (-tol<=diff<=tol)
    while 1:
        diff = midcount-midthr-Margin
        if abs(diff) <= tol:
            break

        # if tolerance is not met up to 20 iteractions, then relax the
        # tolerance by 10#
        iterno += 1

        if iterno>20:
            tol = tol*1.1

        if diff > tol:   # then new bounds are...
            midcount = (upcount+midcount)/2.0
            # upper and middle activities
            midthr = (upthr+midthr)/2.0
            # ... and thresholds
        elif diff < -tol:  # then new bounds are...
            midcount = (midcount+lwcount)/2.0
            # middle and lower activities
            midthr = (midthr+lwthr)/2.0
            # ... and thresholds

    #   Since the tolerance has been satisfied, midcount is selected
    #   as the interpolated value with a tol [dB] tolerance.

    asl_ms_log = midcount
    cc = midthr
    return asl_ms_log, cc
