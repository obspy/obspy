# -*- coding: utf-8 -*-
# 2009-06-10 Moritz
"""
File containing Poles And Zeros (PAZ) and gain for common seismometers.
The instruments must be corrected before to m/s, which is the RESP/SEED
standard.
The seismometer is represented as a dictionary containing the fields:

@type poles: List of Complex Numbers
@ivar poles: Poles of the seismometer to simulate
@type zeros: List of Complex Numbers
@ivar zeros: Zeros of the seismometer to simulate
@type gain: Float
@ivar gain: Gain factor of seismometer to simulate
"""

PAZ_WOOD_ANDERSON = {
    'poles': [-6.2832 - 4.7124j,
              - 6.2832 + 4.7124j],
    'zeros': [0.0 + 0.0j] * 1,
    'gain': 1. / 2.25
}

PAZ_WWSSN_SP = {
    'poles': [-4.0093 - 4.0093j,
              - 4.0093 + 4.0093j,
              - 4.6077 - 6.9967j,
              - 4.6077 + 6.9967j],
    'zeros': [0.0 + 0.0j] * 2,
    'gain': 1. / 1.0413
}

PAZ_WWSSN_LP = {
    'poles': [-0.4189 + 0.0j,
              - 0.4189 + 0.0j,
              - 0.0628 + 0.0j,
              - 0.0628 + 0.0j],
    'zeros': [0.0 + 0.0j] * 2,
    'gain': 1. / 0.0271
}

PAZ_KIRNOS = {
    'poles': [-0.1257 - 0.2177j,
              - 0.1257 + 0.2177j,
              - 83.4473 + 0.0j,
              - 0.3285 + 0.0j],
    'zeros': [0.0 + 0.0j] * 2,
    'gain': 1. / 1.61
}
