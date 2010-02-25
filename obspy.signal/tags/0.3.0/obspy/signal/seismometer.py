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
"""

# Im Wood-Anderson
# findet aber keine Umrechnung statt, es gibt Länge (displacement) als
# Länge (Ausschlag der "Nadel" (Lichtstrahl auf Photoplatte)) wieder aus.
# D.h. diese 2800 sind einfach ein Vergrößerungsfaktor, deine Einheit
# kommt aus dem, worauf du vorher dein anderes Seismometer runtergerechnet
# hast. (thanks to Christian Sippl)
PAZ_WOOD_ANDERSON = {
    'poles': [-6.2832 - 4.7124j,
              - 6.2832 + 4.7124j],
    'zeros': [0.0 + 0.0j] * 1,
    'gain': 2800
}
