package obspy.db
================

Copyright
---------
GNU Lesser General Public License, Version 3 (LGPLv3)

Copyright (c) 2010-2013 by:
    * Robert Barsch


Overview
--------
A seismic waveform indexer and database for ObsPy.

The obspy.db package contains a waveform indexer collecting metadata from a
file based waveform archive and storing in into a standard SQL database.
Supported waveform formats depend on installed ObsPy packages.

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats and
seismological signal processing routines which allow the manipulation of
seismological time series (see Beyreuther et al. 2010, Megies et al. 2011).
The goal of the ObsPy project is to facilitate rapid application development
for seismology.

For more information visit https://www.obspy.org.


Dependencies
------------
* sqlalchemy
