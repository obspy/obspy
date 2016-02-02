package obspy.core
==================

Copyright
---------
GNU Lesser General Public License, Version 3 (LGPLv3)

Copyright (c) 2009-2013 by:
    * Moritz Beyreuther
    * Lion Krischer
    * Robert Barsch
    * Tobias Megies
    * Martin van Driel


Overview
--------
ObsPy - a Python framework for seismological observatories.

The obspy.core package contains common methods and classes for ObsPy required by
all other ObsPy packages. It includes the UTCDateTime, Stats, Stream and Trace
classes and methods for reading and writing seismograms.

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats and
seismological signal processing routines which allow the manipulation of
seismological time series (see Beyreuther et al. 2010, Megies et al. 2011).
The goal of the ObsPy project is to facilitate rapid application development
for seismology.

For more information visit https://www.obspy.org.


Dependencies
------------
* distribute
* NumPy
