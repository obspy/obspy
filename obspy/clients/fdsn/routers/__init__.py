#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.clients.fdsn.routers - FDSN Web Federated catalog service client for ObsPy
======================================================
The obspy.clients.fdsn package contains a client to access web servers that
implement the FDSN web service definitions (https://www.fdsn.org/webservices/).
The holdings of many of these services are indexed by the FedCatalog service
 (https://service.iris.edu/irisws/fedcatalog/1/). Users looking for waveforms or
 metadata can then query the FedCatalog service to learn which provider holds
 which data.  Furthermore, the results from a FedCatalog query are easily turned
 into post requests for each service.

 This FederatedClient first queries the FedCatalog service to determine where
 the data of interest reside.  It then queries the individual web services from
 each provider using the fdsn Client routines to retrieve the resulting data.

:copyright:
    Celso G Reyes, 2017
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Basic Usage
-----------

The first step is always to initialize a client object.

>>> from obspy.clients.fdsn import FederatedClient
>>> client = Client()

Retrieving Station Metadata
---------------------------

Submitting a GET request to the federated catalog service. The service recognizes and accepts 
parameters that are normally accepted by the station web service.

>>> inv = client.get_stations(station="A*", channel="BHZ", level="station")


Retrieving Waveform Metadata
---------------------------
Submitting a GET request to the federated catalog service.  The service recognizes and accepts not
only the paramters normally accepted by the bulkdataselect web service, but also the parameters
accepted by the station service.  This includes geographic parameters. For more details, see the
help for obspy.clients.fdsn

    >>> from obspy import UTCDateTime
    >>> t = UTCDateTime("2010-02-27T06:45:00.000")
    >>> st = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
    >>> st.plot()  # doctest: +SKIP

    .. plot::

        from obspy import UTCDateTime
        from obspy.clients.fdsn import FederatedClient
        client = Client()
        t = UTCDateTime("2010-02-27T06:45:00.000")
        st = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
        st.plot()

caveats
-------
1. duplicated metadata remains duplicated.

"""

# convenience imports
from .routing_client import (RoutingClient, ResponseManager)  # NOQA
from .routing_response import RoutingResponse    # NOQA
from .fedcatalog_response_parser import (FederatedResponse, FedcatalogProviderMetadata)    # NOQA
from .fedcatalog_routing_client import (FederatedClient,
                                        FederatedResponseManager)    # NOQA
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
