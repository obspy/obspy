package obspy.clients.neic
==========================

Copyright
---------
GNU Lesser General Public License, Version 3 (LGPLv3)

Copyright (c) 2013 by:
    * David Ketchum


Overview
--------
NEIC CWB Query service client for ObsPy.

The obspy.clients.neic package contains a client for the QueryServer software
run at the National Earthquake Information Center (NEIC) in Golden, CO USA.
This service is run publicly at cwbpub.cr.usgs.gov on port 2061.  This server
returns data based on a query command string as a series of binary MiniSEED
blocks.  The data are not necessarily in order and may have gaps, overlaps and
possibly duplicate blocks.  The client (this software) has to deal with these
irregularities. There is a java based client CWBQuery available at
ftp://hazards.cr.usgs.gov/CWBQuery which implements this protocol and has a
variety of output formats and "cleanup" modes.

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats and
seismological signal processing routines which allow the manipulation of
seismological time series (see Beyreuther et al. 2010, Megies et al. 2011).
The goal of the ObsPy project is to facilitate rapid application development
for seismology.

For more information visit https://www.obspy.org.
