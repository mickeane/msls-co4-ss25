import numpy as np
from scipy.fft import rfft, rfftfreq, irfft


# Code from: https://stackoverflow.com/a/67127726/3388962
# The below makes use of a decorator to generate a function that 
# takes a single argument N and returns a signal with a given 
# power spectral density (PSD).
def noise_psd(N, psd = lambda f: 1):
        X_white = rfft(np.random.randn(N));
        S = psd(rfftfreq(N))
        # Normalize S
        S = S / np.sqrt(np.mean(S**2))
        X_shaped = X_white * S;
        return irfft(X_shaped);

def PSDGenerator(f):
    return lambda N: noise_psd(N, f)

@PSDGenerator
def white_noise(f):
    return 1;

@PSDGenerator
def blue_noise(f):
    return np.sqrt(f);

@PSDGenerator
def violet_noise(f):
    return f;

@PSDGenerator
def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))