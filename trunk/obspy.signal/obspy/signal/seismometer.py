# -*- coding: utf-8 -*-
"""
File containing Poles And Zeros (PAZ) and gain for common seismometers.
The instruments must be corrected before to velocity, which is the RESP/SEED
standard.

The seismometer is represented as a dictionary containing the fields:

:type poles: List of Complex Numbers
:ivar poles: Poles of the seismometer to simulate
:type zeros: List of Complex Numbers
:ivar zeros: Zeros of the seismometer to simulate
:type gain: Float
:ivar gain: Gain factor of seismometer to simulate
:type sensitivity: Float
:ivar sensitivity: Overall sensitivity of seismometer to simulate

Currently contained seismometers::

* PAZ_WOOD_ANDERSON
* PAZ_WWSSN_SP
* PAZ_WWSSN_LP
* PAZ_KIRNOS
"""

# Seismometers defined as in Pitsa with one zero less. The corrected
# signals are in velocity, thus must be integrated to offset and take one
# zero less than pitsa (remove 1/w in frequency domain)

PAZ_WOOD_ANDERSON = {'poles': [-6.2832 - 4.7124j,
                               -6.2832 + 4.7124j],
                     'zeros': [0.0 + 0.0j] * 1,
                     'sensitivity': 1.0,
                     'gain': 1. / 2.25}

PAZ_WWSSN_SP = {'poles': [-4.0093 - 4.0093j,
                          -4.0093 + 4.0093j,
                          -4.6077 - 6.9967j,
                          -4.6077 + 6.9967j],
                'zeros': [0.0 + 0.0j] * 2,
                'sensitivity': 1.0,
                'gain': 1. / 1.0413}

PAZ_WWSSN_LP = {'poles': [-0.4189 + 0.0j,
                          -0.4189 + 0.0j,
                          -0.0628 + 0.0j,
                          -0.0628 + 0.0j],
                'zeros': [0.0 + 0.0j] * 2,
                'sensitivity': 1.0,
                'gain': 1. / 0.0271}

PAZ_KIRNOS = {'poles': [-0.1257 - 0.2177j,
                        -0.1257 + 0.2177j,
                        -83.4473 + 0.0j,
                        -0.3285 + 0.0j],
              'zeros': [0.0 + 0.0j] * 2,
              'sensitivity': 1.0,
              'gain': 1. / 1.61}

INSTRUMENTS = {'None': None,
               'kirnos': PAZ_KIRNOS,
               'wood_anderson': PAZ_WOOD_ANDERSON,
               'wwssn_lp': PAZ_WWSSN_LP,
               'wwssn_sp': PAZ_WWSSN_SP}
