import numpy as np

import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing
from scipy.signal import savgol_filter

# Load sample data.
st = read()
tr = st[0].data

# Get the amplitude spectra and corresponding frequencies.
amp_spec = np.abs(np.fft.rfft(tr.data))
freqs = np.fft.rfftfreq(len(tr.data), 1. / tr.stats.sampling_rate)

# Apply Konno-Ohmachi smoothing
konno_smooth = konno_ohmachi_smoothing(amp_spec[0], freqs, normalize=True)

# Apply a Savitzky-Golay filter (a linear space smoother).
savitzky_smooth = savgol_filter(amp_spec, 51, 3)

# Plot using matplotlib.
plt.loglog(freqs, amp_spec, label='raw data', c='0.5', alpha=.6)
plt.loglog(freqs, savitzky_smooth, label='Savitzky Golay smoothing')
plt.loglog(freqs, konno_smooth, label='Konno Ohmachi smoothing')
plt.xlabel('frequency')
plt.ylabel('amplitude')
plt.legend()
plt.show()
