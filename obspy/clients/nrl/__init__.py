# -*- coding: utf-8 -*-
"""
obspy.clients.nrl - Nominal Response Library client for ObsPy
=============================================================

This module contains a client to access the `IRIS Library of Nominal Response
for Seismic Instruments <https://ds.iris.edu/NRL/>`_ (NRL).
To cite use of the NRL, please see [Templeton2017]_.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Basic Usage
-----------

.. warning::
    Connecting to NRL hosted online is deprecated. The new NRLv2 will stop
    providing navigational information in machine readable form in favor of the
    html navigation, so the existing client for online use will stop working
    when the original NRLv1 is taken offline (announced for Spring 2023).
    Please consider using a full downloaded copy of the NRL (v1 or v2,
    `instructions on NRL homepage <https://ds.iris.edu/ds/nrl/>`_) providing a
    local path, e.g. ``nrl = NRL('./downloads/NRL')``.

The first step is to initialize a NRL client object. A client object can be
initialized either with the base URL of a NRL hosted on a web server or with a
local directory path to a downloaded and unpacked NRL zip file. The default is
to access the always up-to-date NRL database hosted at IRIS.

>>> from obspy.clients.nrl import NRL
>>> nrl = NRL()

The sensor and datalogger tree structure can be interactively explored in an
interactive Python shell:

>>> print(nrl) # doctest: +SKIP
NRL library at http://ds.iris.edu/NRL/
  Sensors: 24 manufacturers
    'CEA-DASE', 'CME', 'Chaparral Physics', 'Eentec', 'GEObit',
    'GEOsig', 'Generic', 'Geo Space/OYO', 'Geodevice', 'Geotech',
    'Guralp', 'Hyperion', 'IESE', 'Kinemetrics', 'LaHusen', 'Lennartz'
    'Metrozet', 'Nanometrics', 'REF TEK', 'Sercel/Mark Products',
    'Silicon Audio', 'SolGeo', 'Sprengnether (now Eentec)',
    'Streckeisen'
  Dataloggers: 15 manufacturers
    'Agecodagis', 'DAQ Systems (NetDAS)', 'Earth Data', 'Eentec',
    'Generic', 'GeoBIT', 'Geodevice', 'Geotech', 'Guralp',
    'Kinemetrics', 'Nanometrics', 'Omnirecs', 'Quanterra', 'REF TEK',
    'SolGeo'
>>> print(nrl.sensors) # doctest: +SKIP
Select the sensor manufacturer (24 items):
  'CEA-DASE', 'CME', 'Chaparral Physics', 'Eentec', 'GEObit', 'GEOsig'
  'Generic', 'Geo Space/OYO', 'Geodevice', 'Geotech', 'Guralp',
  'Hyperion', 'IESE', 'Kinemetrics', 'LaHusen', 'Lennartz', 'Metrozet'
  'Nanometrics', 'REF TEK', 'Sercel/Mark Products', 'Silicon Audio'
  'SolGeo', 'Sprengnether (now Eentec)', 'Streckeisen'
>>> print(nrl.sensors['Streckeisen']) # doctest: +SKIP
Select the Streckeisen sensor model (5 items):
  'STS-1', 'STS-2', 'STS-2.5', 'STS-3', 'STS-5A'
>>> print(nrl.sensors['Streckeisen']['STS-1']) # doctest: +SKIP
Select the corner period mode for this STS-1 (2 items):
  '20 seconds', '360 seconds'
>>> print(nrl.sensors['Streckeisen']['STS-1']['360 seconds']) # doctest: +SKIP
(u'STS-1, 360 s mode, 2400 V/m/s',
 u'http://ds.iris.edu/NRL/sensors/streckeisen/RESP.XX.NS088..BHZ.STS1.360.2400')  # NOQA

Response objects can be extracted by providing the datalogger and sensor keys:

>>> response = nrl.get_response( # doctest: +SKIP
...     sensor_keys=['Streckeisen', 'STS-1', '360 seconds'],
...     datalogger_keys=['REF TEK', 'RT 130 & 130-SMA', '1', '200'])
>>> print(response) # doctest: +SKIP
Channel Response
    From M/S (Velocity in Meters per Second) to COUNTS (Digital Counts)
    Overall Sensitivity: 1.50991e+09 defined at 0.020 Hz
    10 stages:
        Stage 1: PolesZerosResponseStage from M/S to V, gain: 2400
        Stage 2: ResponseStage from V to V, gain: 1
        Stage 3: CoefficientsTypeResponseStage from V to COUNTS, gain: 629129
        Stage 4: CoefficientsTypeResponseStage from COUNTS to COUNTS, gain: 1
        Stage 5: CoefficientsTypeResponseStage from COUNTS to COUNTS, gain: 1
        Stage 6: CoefficientsTypeResponseStage from COUNTS to COUNTS, gain: 1
        Stage 7: CoefficientsTypeResponseStage from COUNTS to COUNTS, gain: 1
        Stage 8: CoefficientsTypeResponseStage from COUNTS to COUNTS, gain: 1
        Stage 9: CoefficientsTypeResponseStage from COUNTS to COUNTS, gain: 1
        Stage 10: CoefficientsTypeResponseStage from COUNTS to COUNTS, gain: 1
"""
from .client import NRL
