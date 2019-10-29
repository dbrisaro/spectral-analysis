#!/usr/bin/env python3

"""
this function returns the theoretical red noise
PSD from a signal using DFT

Dani Risaro, adapted from Matias Risaro
Octubre 2019
#Volvimos
"""

def psd_rednoise(signal):
    import numpy as np
    from scipy import optimize
    import scipy.signal

    def filt(x, a, b, c):                       # fitting function
         return a*(1/(1+(x/b)**c))

    def log_func(x, a, b, c):                   # ... and the log of it
       return np.log(filt(x, a, b, c))

    nfft = len(signal)
    t = np.arange(nfft)
    sp = scipy.fftpack.fft(signal, nfft)/nfft
    freq = np.fft.fftfreq(t.shape[-1])
    qq, = np.where(freq>0)

    xdft_short = sp[qq]

    Pxx = (np.abs(xdft_short))**2 # power in each bin.
    Pxx_density = Pxx / (1/nfft)  # power is energy over -fs/2 to fs/2, with nfft bins
    Pxx_density[1:-1] = 2*Pxx_density[1:-1] # conserve power since we threw away 1/2 the spectrum
    # note that DC (0 frequency) and Nyquist term only appear once, we don't double those.

    S_pow = Pxx_density
    f = freq[qq]
    # Fitting de function filt
    popt, pcov = optimize.curve_fit(log_func, f, np.log(S_pow))

    # simulate red noise
    f_noise = f
    P_noise = filt(f_noise,popt[0],popt[1],popt[2])

    return f_noise, P_noise
