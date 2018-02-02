from obspy.core.util.base import get_example_file
from obspy.signal import PPSD


ppsd = PPSD.load_npz(get_example_file('ppsd_kw1_ehz.npz'))
ppsd.plot_spectrogram()
