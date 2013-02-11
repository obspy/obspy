"""
DEPRECATED: Use obspy.signal.spectral_estimation instead.

Various Routines Related to Spectral Estimation

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import warnings
from obspy.signal.spectral_estimation import psd, fft_taper, welch_taper, \
        welch_window, PPSD, get_NLNM, get_NHNM

msg = 'Module obspy.signal.psd is deprecated! Use ' + \
      'obspy.signal.spectral_estimation instead or import directly ' + \
      '"from obspy.signal import ...".'
warnings.warn(msg, category=DeprecationWarning)
