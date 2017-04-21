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
from .fedcatalog_parser import (FederatedResponse)  # NOQA
from .fedcatalog_client import (FederatedClient, FederatedResponseManager, 
                                FedcatalogProviders)  # NOQA
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)

'''
client = FederatedClient([**individualClientArgs])
      -> FederatedClient [no init function] 
      -> RoutingClient.__init__(use_parallel, **kwargs)

inv = client.get_stations([argsToFederator][argsForEachRequest])
      -> FederatedClient.get_stations(exclude, include, includeoverlaps, **kwargs)
      -> query_fedcatalog(SERVICE, params, bulk) ... returns Response
      -> FederatedResponseManager.__init__(resp.txt, include, exclude)
      ->   ResponseManager.__init(text, include, exclude)
               FRM.parse_response(text) OR save into .responses
               .subset_requests(include, exclude)
               [now, FRM.Responses filled with FederatedResponse items]

      -> querysvc = RoutingClient.get_query_machine()
            [returns either RoutingClient.serial_service_query or RoutingClient.parallel_service_query]

      -> data,retry = querysvc(FRM, SERVICE, [**kwargs]) 
         [eg. RoutingClient.serial_service_query(FRM, SERVICE,[argsForEachRequest])]
              -> set up output, failed queues
              -> FOR EACH REQ in FRM
                -> client = fdsn.client.Client(REQ.code, FedClient.individualClientArgs)
                -> fn = FederatedClient.get_request_fn(targetservice)
                        [either FC.submit_waveform_request or FC.submit_station_request]
                -> fn(client, request, output, failed, [argsForEachRequest])
                    [FC.submit_station/waveform_request(client,req,out,fai,**kwarg)]
                    -> client.get_stations/waveforms_bulk(bulk=req.text(SERVICE), **kwargs)
                       [all or nothing... either data goes to output.put or requestlines go to failed.put]
         [returns data, retry]
'''
