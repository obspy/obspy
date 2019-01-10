import numpy as np

from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing

random = np.random.RandomState(13)

# Generate 20 minutes of random data at 100 Hz with 3 components.
sampling_rate = 100
number_of_components = 3
seconds = 1200
data = random.rand(number_of_components, sampling_rate * seconds)

# Get the amplitude spectra and corresponding frequencies.
amp_spec = np.abs(np.fft.rfft(data))
freqs = np.fft.rfftfreq(data.shape[-1], 1. / sampling_rate)

# Define a subset of frequencies which are of interest.
center_freqs = np.logspace(-3, 1, num=200)

# Apply smoothing on all three channels.
konno_smooth = konno_ohmachi_smoothing(
    amp_spec, freqs, normalize=True, center_frequencies=center_freqs
)

print(konno_smooth.shape)
