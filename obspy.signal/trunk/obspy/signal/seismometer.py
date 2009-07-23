# -*- coding: utf-8 -*-
# 2009-06-10 Moritz
"""
File containing Poles And Zeros (PAZ) and gain for common seismometers to simulate.
The seismometer is represented as a dictionary containing the fields:

@type poles: List of Complex Numbers
@ivar poles: Poles of the seismometer to simulate
@type zeroes: List of Complex Numbers
@ivar zeroes: Zeroes of the seismometer to simulate
@type gain: Float
@ivar gain: Gain factor of seismometer to simulate
"""

wood_anderson = {
    'poles' : [-6.2832 - 4.7124j,
               -6.2832 + 4.7124j],
    'zeroes': [0.0 + 0.0j,
               0.0 + 0.0j],
    'gain': 1./2.25
}

wwssn_sp = {
    'poles' : [-4.0093 - 4.0093j,
               -4.0093 + 4.0093j,
               -4.6077 - 6.9967j,
               -4.6077 + 6.9967j],
    'zeroes': [0.0 + 0.0j,
               0.0 + 0.0j,
               0.0 + 0.0j],
    'gain': 1./1.0413
}

wwssn_lp = {
    'poles' : [-0.4189 + 0.0j,
               -0.4189 + 0.0j,
               -0.0628 + 0.0j,
               -0.0628 + 0.0j],
    'zeroes': [0.0 + 0.0j,
               0.0 + 0.0j,
               0.0 + 0.0j],
    'gain': 1./0.0271
}

kirnos = {
    'poles' : [-0.1257 - 0.2177j,
               -0.1257 + 0.2177j,
               -83.4473 + 0.0j,
               -0.3285 + 0.0j],
    'zeroes': [0.0 + 0.0j,
               0.0 + 0.0j,
               0.0 + 0.0j],
    'gain': 1./1.61
}
