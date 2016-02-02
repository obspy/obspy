package obspy.io.mseed
======================

Copyright
---------
GNU Lesser General Public License, Version 3 (LGPLv3)

Copyright (c) 2009-2013 by:
    * Lion Krischer
    * Robert Barsch
    * Moritz Beyreuther
    * Chad Trabant


Overview
--------
Mini-SEED read and write support for ObsPy.

This module provides read and write support for Mini-SEED waveform data and
some other convenient methods to handle Mini-SEED files. Most methods are based
on libmseed, a C library framework by Chad Trabant and interfaced via python
ctypes.

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats and
seismological signal processing routines which allow the manipulation of
seismological time series (see Beyreuther et al. 2010, Megies et al. 2011).
The goal of the ObsPy project is to facilitate rapid application development
for seismology.

For more information visit https://www.obspy.org.
