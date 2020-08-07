package obspy.io.gse2
=====================

Copyright
---------
GNU Lesser General Public License, Version 3 (LGPLv3)

Copyright (c) 2009-2013 by:
    * Moritz Beyreuther
    * Stefan Stange
    * Robert Barsch


Overview
--------
GSE2 and GSE1 read and write support for ObsPy.

This module provides read and write support for GSE2 CM6 compressed as well as
GSE1 ASCII waveform data and header info. Most methods are based on the C
library GSE_UTI of Stefan Stange, which is interfaced via Python ctypes.

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats and
seismological signal processing routines which allow the manipulation of
seismological time series (see Beyreuther et al. 2010, Megies et al. 2011).
The goal of the ObsPy project is to facilitate rapid application development
for seismology.

For more information visit https://www.obspy.org.
