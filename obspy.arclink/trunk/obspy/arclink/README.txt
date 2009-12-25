package obspy.fissures
======================

Copyright
---------
    GNU Lesser General Public License, Version 3 (LGPLv3)

    Copyright (c) 2009-2010 by:
        * Robert Barsch


Overview
--------
    obspy.fissures - DHI/Fissures request client for of ObsPy.
    
    The Data Handling Interface (DHI) is a CORBA data access framework
    allowing users to access seismic data and metadata from IRIS DMC
    and other participating institutions directly from a DHI-supporting
    client program. The effect is to eliminate the extra steps of
    running separate query interfaces and downloading of data before
    visualization and processing can occur. The information is loaded
    directly into the application for immediate use.
    http://www.iris.edu/dhi/
    
    Detailed information on network_dc and seismogram_dc servers:
     * http://www.seis.sc.edu/wily
     * http://www.iris.edu/dhi/servers.htm


Dependencies
------------
    * setuptools
    * obspy.core
    * obspy.mseed
    * omniORB 
    