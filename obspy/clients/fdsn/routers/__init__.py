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
 into POST requests for each service.

 This FederatedClient first queries the FedCatalog service to determine where
 the data of interest reside.  It then queries the individual web services from
 each provider using the fdsn Client routines to retrieve the resulting data.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
    Celso G Reyes, 2017
    IRIS-DMC
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Basic Usage
-----------

The first step is always to initialize a client object.

>>> from obspy.clients.fdsn import FederatedClient
>>> client = FederatedClient()

Retrieving Station Metadata
---------------------------

Submitting a GET request to the federated catalog service. The service recognizes
parameters that are normally accepted by the station web service.

>>> inv = client.get_stations(station="A*", channel="BHZ", level="station")


Retrieving Waveform Metadata
---------------------------
Submitting a GET request to the federated catalog service.  The service recognizes
not only the paramters normally accepted by the bulkdataselect web service, but
also the parameters accepted by the station service.  This includes geographic
parameters. For more details, see the help for obspy.clients.fdsn

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
----
* duplicated metadata remains duplicated.

suggestions for future
----------------------
* bulk requests for a certain level of metadata, eg. network, station
  will return an egregious amount of data that is very likely duplicated
  Perhaps the fedcatalog web service could recognize it and return pared
  down data or the fdsn.client.Client  class could recognize and reduce
  the duplications. example:
    >>> INV2 = fclient.get_stations(network="I?", station="AN*",
    ...                           level="network", includeoverlaps="true")
    ...                           #doctest: +SKIP

"""

# convenience imports
from .routing_client import (RoutingClient, RoutingManager)
from .fedcatalog_parser import (FederatedRoute)  # NOQA
from .fedcatalog_client import (FederatedClient, FederatedRoutingManager,
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
      -> get_routing(SERVICE, params, bulk) ... returns Response
      -> FederatedRoutingManager.__init__(resp.txt, include, exclude)
      ->   RoutingManager.__init(text, include, exclude)
               FRM.parse_routing(text) OR save into .routes
               [now, FRM.Responses filled with FederatedRoute items]

      -> querysvc = RoutingClient.get_query_machine()
            [returns either RoutingClient.serial_service_query or RoutingClient.parallel_service_query]

      -> data,retry = querysvc(FRM, SERVICE, [**kwargs]) 
         [eg. RoutingClient.serial_service_query(FRM, SERVICE,[argsForEachRequest])]
              -> set up output, failed queues
              -> FOR EACH REQ in FRM
                -> client = fdsn.client.Client(REQ.provider_id, FedClient.individualClientArgs)
                -> fn = FederatedClient.get_request_fn(targetservice)
                        [either FC.submit_waveform_request or FC.submit_station_request]
                -> fn(client, request, output, passed, failed, [argsForEachRequest])
                    [FC.submit_station/waveform_request(client,route,out,fai,**kwarg)]
                    -> client.get_stations/waveforms_bulk(bulk=route.text(SERVICE), **kwargs)
                       [all or nothing... either data goes to output.put or requestlines go to failed.put]
         [returns data, retry]
'''
