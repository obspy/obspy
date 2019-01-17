import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing

random = np.random.RandomState(13)

# Generate 20 minutes of random data at 100 Hz with 3 components.
sampling_rate = 100
number_of_components = 3
seconds = 20 * 60
data = random.rand(number_of_components, sampling_rate * seconds)

# Detrend the data
detrended_data = scipy.signal.detrend(data, type='linear')

# Get the amplitude spectra and corresponding frequencies.
spectra = np.abs(np.fft.rfft(detrended_data))
freqs = np.fft.rfftfreq(data.shape[-1], 1. / sampling_rate)

# Define a subset of frequencies which are of interest.
center_freqs = np.logspace(-2, 1, num=250)

# Apply smoothing on all three channels.
smooth_spectra = konno_ohmachi_smoothing(
    spectra, freqs, normalize=True, center_frequencies=center_freqs
)

# Iterate and plot each smoothed spectrum.
for smooth_spectrum in smooth_spectra:
    plt.loglog(center_freqs, smooth_spectrum)
plt.xlabel('frequency')
plt.ylabel('amplitude')
plt.show()
