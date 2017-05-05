#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.clients.fdsn.routers - FDSN Federated catalog webservice client for ObsPy
======================================================
The obspy.clients.fdsn package contains a client to access web servers that
implement the FDSN web service definitions (https://www.fdsn.org/webservices/).
The holdings of many of these services are indexed by the FedCatalog service
 (https://service.iris.edu/irisws/fedcatalog/1/). Users searching for waveforms
 or metadata can then query the FedCatalog service to learn which provider
 holds which data. Furthermore, the results from a FedCatalog query are easily
 turned into POST requests for each service.

 This FederatedClient first queries the FedCatalog service to determine where
 the data of interest reside. It then queries the individual web services from
 each provider using the FDSN Client routines to retrieve the resulting data.

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

Submitting a GET request to the federated catalog service. The service
recognizes parameters that are normally accepted by the station web service.

>>> inv = client.get_stations(station="A*", channel="BHZ", level="station")


Retrieving Waveform Metadata
---------------------------
Submitting a GET request to the federated catalog service. The service
recognizes not only the parameters normally accepted by the bulkdataselect web
service, but also the parameters accepted by the station service. This
includes geographic parameters. For more details, see the help for
obspy.clients.fdsn

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


Getting comfortable with the building blocks of the routing modules:
The simplest unit of a request is the FDSNBulkRequestItem.

FDSNBulkRequestItem
------------------------------------------------------
The FDSNBulkRequestItem represents one bulk request line, containing the
following information:

  network station location channel starttime endtime

in a format that might look like A:

A)  IU ANMO 00 BHZ 2015-05-24T12:00:00 2015-05-24T12:05:00

It could include wildcards in any of the fields, as shown in B...

B)  IU ANMO * * 2015-05-24T12:00:00 *

Using obspy.clients.fdsn.routers.FDSNBulkRequestItem, each could be created as:
itemA = FDSNBulkRequestItem('IU ANMO 00 BHZ 2015-05-24T12:00:00 2015-05-24T12:05:00')

And, item B, with wildcards could be created similarly.
>>> itemB = FDSNBulkRequestItem('IU ANMO * * 2015-05-24T12:00:00 *')

  or, with parameters...
>>> itemB = FDSNBulkRequestItem(network='IU', station='ANMO',
...                             starttime='2015-05-24T12:00:00')

  or, using obspy.core.UTCDateTime ...
>>> t1 = UTCDateTime(2015,05,24,12,0,0)
>>> itemB = FDSNBulkRequestItem(network='IU', station='ANMO', starttime=t1)

Basic comparisons can be made between these items.
A < B : alphabetically compares net, stations, locations, then chans,
        and then starttime by date.
A contains B : This takes wildcards and time ranges into account to denote
               whether a request for A would include B's data too.
A == B : all fields are the same (wildcards only match wildcards, for example)

FDSNBulkRequestItems are grouped into FDSNBulkRequests
----------------------------------------------------
Typically, though, several lines are sent together in a bulk request. These are
handled by FDSNBulkRequests. Here is an example of a few ways to create these:

>>> txt= '''
  IU ANMO 00 BHZ 2015-05-24T12:00:00 2015-05-24T12:05:00
  IU ANMO 10 BHZ 2015-05-24T12:00:00 2015-05-24T12:05:00
  IU ANTO 00 BHZ 2015-05-24T12:00:00 2015-05-24T12:05:00
  '''
Each line of txt is converted to an FDSNBulkRequestItem
>>> brq = FDSNBulkRequests(txt)

remember, these are FDSNBulkRequestItems
>>> brq = FDSNBulkRequests([itemA, itemB])

A couple properties of FDSNBulkRequests:
1. FDSNBulkRequests store items in a set, which means:
   A. duplicate items are ignored
   B. they can be combined with another set (via `update()`
   C. swaths of requests can be removed via `difference_update()`
2. string representations are in sorted order (Net, Sta, Loc, Cha, start time).
   End time is ignored.

FDSNBulkRequests can ALSO be created from obspy data, via a conversion routine
 `data_to_request()` which can convert Inventory and Stream items.
- Stream items will be converted directly, 1:1 to FDSNBulkRequestItems
  within the FDSNBulkRequest.
- Inventory items are converted according to their level. So, an inventory tree
  that contains only network and station data would have wildcards for the
  location and channel. Additionally, the start and end-time will be those for
  the station, which might not reflect channel, data, or network start and end
  times.

ASKING FOR DATA
===============
The fdsn.FederatedClient is accessed in much the same way as the fdsn.Client.
The primary methods for getting data include the standard four:
    get_stations(), get_stations_bulk(), get_waveforms(), get_waveforms_bulk()
additionally, there are two methods for retrieving raw Fedcatalog data, which
do not attempt to further contact data providers:
    get_routing(), get_routing_bulk()

When retrieving data through one of the typical fdsn.Client methods, the
routing service (eg. Fedcatalog) is first queried with either parameters
or a bulk request. It then responds with a file that is, essentially, a series
of bulk requests. There will be one section for each provider (datacenter) that
specifies the provider, lists its services, and then lists the bulk-request
to be sent to that provider. Included at the top of this file, one will find
the param=value pairs that help specify this particular request. For example,
a station request might have level=channel at the top.

These are parsed, and each one is converted to a RoutingResponse. Each series
of bulk requests are turned into a FDSNBulkRequests object.

The entire reply from the routing service is encapsulated into a RoutingManager
(really, a FederatedRoutingManager). That is, the RoutingManager contains and
manages several RoutingResponses.

There are two paths that may be followed to get data from the clients. The
FederatedClient can either loop through and request data from the provider
one after the other (serial query), or may send out requests to all clients,
and then reassemble the data once it is retrieved. If the requested data
is not retrieved, then the corresponding BulkRequestItems could be rerouted.

The FederatedClient.attempt_reroute() queries the Fedcatalog for all possible
places where the missing data could be found, and then requeries each provider.
With any luck, the desired data could be downloaded from elsewhere, and
was added to the retrieved data.
"""

# convenience imports
from .routing_client import (RoutingClient, RoutingManager)  # NOQA
from .fedcatalog_parser import (FederatedRoute)  # NOQA
from .fedcatalog_client import (FederatedClient, FederatedRoutingManager,  # NOQA
                                FedcatalogProviders)  # NOQA
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
