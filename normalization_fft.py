# FFT normalization to conserve power

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

s = np.loadtxt('/home/daniu/Documentos/test.out')

sample_rate = 1
time_step = 1/sample_rate

carrier_I = s
t = np.arange(len(s))
num_samples = len(s)

#######################################################
# FFT using Welch method
# windows = np.ones(nfft) - no windowing
# if windows = 'hamming', etc.. this function will
# normalize to an equivalent noise bandwidth (ENBW)
#######################################################
nfft = num_samples  # fft size same as signal size
f,Pxx_den = scipy.signal.welch(carrier_I, fs = 1/time_step,\
                    window = np.ones(nfft),\
                    nperseg = nfft,\
                    scaling='density', detrend='linear')

nperseg  = nfft/2
noverlap = nperseg*(3/4)
f, Pxx_den = scipy.signal.welch(carrier_I, fs=1, window="hanning",
                nperseg=int(nperseg), noverlap=int(noverlap),
                scaling='density', detrend='linear')

###############################################################################
# FFT comparison
###############################################################################

integration_time = nfft*time_step
power_time_domain = sum((np.abs(carrier_I)**2)*time_step)/integration_time
print('power time domain = %f' % power_time_domain)

# Take FFT.  Note that the factor of 1/nfft is sometimes omitted in some
# references and software packages.
# By proving Parseval's theorem (conservation of energy) we can find out the
# proper normalization.
signal = carrier_I
xdft = scipy.fftpack.fft(signal, nfft)/nfft
# fft coefficients need to be scaled by fft size
# equivalent to scaling over frequency bins
# total power in frequency domain should equal total power in time domain
power_freq_domain = sum(np.abs(xdft)**2)
print('power frequency domain = %f' % power_freq_domain)
# Energy is conserved

xdft_short = xdft[0:int(nfft/2+1)] # take only positive frequency terms, other half identical
# xdft[0] is the dc term
# xdft[nfft/2] is the Nyquist term, note that Python 2.X indexing does NOT
# include the last element, therefore we need to use 0:nfft/2+1 to have an array
# that is from 0 to nfft/2
# xdft[nfft/2-x] = conjugate(xdft[nfft/2+x])
Pxx = (np.abs(xdft_short))**2 # power ~ voltage squared, power in each bin.
Pxx_density = Pxx / (sample_rate/nfft)  # power is energy over -fs/2 to fs/2, with nfft bins
Pxx_density[1:-1] = 2*Pxx_density[1:-1] # conserve power since we threw away 1/2 the spectrum
# note that DC (0 frequency) and Nyquist term only appear once, we don't double those.
# Note that Python 2.X array indexing is not inclusive of the last element.

freq = np.linspace(0,sample_rate/2,nfft/2+1)
# frequency range of the fft spans from DC (0 Hz) to
# Nyquist (Fs/2).
# the resolution of the FFT is 1/t_stop
# dft of size nfft will give nfft points at frequencies
# (1/stop) to (nfft/2)*(1/t_stop)

plt.figure(1)
plt.title('Variance preserving plot of the power spectral density (PSD)')
plt.semilogx(freq, Pxx_density*freq, '^-')
plt.figure(1)
plt.semilogx(f, Pxx_den*f)
plt.xlabel('Freq [cycles mo$^{-1}$]'),plt.ylabel('f*PSD [$^{\circ}$C$^2$/cycles mo$^{-1}$]')
plt.show()
