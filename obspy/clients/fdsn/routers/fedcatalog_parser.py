#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains request-related classes, with supporting parsing routines

Request-related classes
-----------------------
:class:`~obspy.clients.fdsn.routers.FDSNBulkRequestItem`
Contains one request's details. The format is STA NET LOC CHA STARTTIME ENDTIME
It contains basic functionality allowing it to be hashed, compared, and printed

:class:`~obspy.clients.fdsn.routers.FDSNBulkRequests`
Handles a set of :class:`~obspy.clients.fdsn.routers.FDSNBulkRequestItem`,
allowing them to be retrieved as a big text string, and providing the ability
to do set comparisons.

:class:`~obspy.clients.fdsn.routers.RoutingResponse`
Serves as an abstract base for routes.  A route is the combination of a data
provider along with multiple specific requests.

:class:`~obspy.clients.fdsn.routers.FederatedRoute`
Route devoted to an FDSN provider.  It contains the who, what, and where
needed to send requests to a service, and provides the facilities for
adding service endpoints, request items, and additional query parameters.
Requests are stored as :class:`~obspy.clients.fdsn.routers.FDSNBulkRequests`

Parsing-related classes
-----------------------
Classes related to the parsing of the Fedcatalog response:
The base class is :class:`~obspy.clients.fdsn.routers.ParserState`
The other classes inherit from the base class, and are specialized to handle
one particular line from the response.  These classes are:
:class:`~obspy.clients.fdsn.routers.PreParse`
:class:`~obspy.clients.fdsn.routers.ParameterItem`
:class:`~obspy.clients.fdsn.routers.EmptyItem`
:class:`~obspy.clients.fdsn.routers.DataCenterItem`
:class:`~obspy.clients.fdsn.routers.ServiceItem`
:class:`~obspy.clients.fdsn.routers.RequestItem`

parsing is assisted by :class:`~obspy.clients.fdsn.routers.FedcatResponseLine`

Misc. Functions
---------------
Additional functions in this file provide mechanisms to represent items from
:class:`~obspy.core.Stream` and :class:`~obspy.core.inventory.Inventory`
as FDSNBulkRequests

:func:`inventory_to_bulkrequests()`
:func:`stream_to_bulkrequests()`

:copyright:
    The ObsPy Development Team (devs@obspy.org)
    Celso G Reyes, 2017
    IRIS-DMC
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from future.utils import string_types

import sys
import collections
from obspy.core import UTCDateTime


class FDSNBulkRequestItem(object):
    """
    representation of a bulk request line containing a single request's details

    The format is STA NET LOC CHA STARTTIME ENDTIME
    It contains functionality allowing it to be hashed, compared, and printed

    >>> line = "IU ANMO 00 BHZ 2012-05-06T12:00:00 2012-05-06T13:00:00"
    >>> FDSNBulkRequestItem(line=line)
    IU ANMO 00 BHZ 2012-05-06T12:00:00.000 2012-05-06T13:00:00.000

    >>> A = FDSNBulkRequestItem(line=line)
    >>> B = FDSNBulkRequestItem(network="IU", station="ANTO", channel="BHZ")
    >>> B
    IU ANTO * BHZ * *
    >>> A < B
    True
    >>> C = FDSNBulkRequestItem(network="IU", station="ANTO", channel="*")
    >>> A.contains(C), B.contains(C), C.contains(B)
    (False, False, True)
    >>> (A < B, B < C)
    (True, False)
    >>> A == FDSNBulkRequestItem(line=line)
    True
    """
    def __init__(self, line=None, network=None, station=None, location=None,
                 channel=None, starttime=None, endtime=None, **kwargs):
        """
        Smart way to handle bulk request lines, for easy comparisons
        specifying a whole line overrides any individual choices

        :type line: str
        :param line: space-delimited, 6-field long bulk request line with the
        following format:  "network station location channel starttime endtime"
        where each field conforms to SEED conventions.  Ex:
            'IU ANMO 00 BHZ 2010-02-27T06:30:00.000 2010-02-27-T10:30:00.000'
        simple wildcards are accepted: '?', '*'
        See also: https://service.iris.edu/fdsnws/dataselect/docs/1/help/
        :type network: str
        :param network: network code, usually 2 letters.
        :type station: str
        :param station: station code
        :type location: str
        :param location: two letter/digit location code.
            Use '--' for blank (ie, '  ')
        :type channel: str
        :param channel: SEED channel code
        :type starttime: str or UTCDateTime
        :param starttime: start time of miniSEED data or limit to metadata
            listed on or after this date
        :type: str or UTCDateTime
        :param endtime: end time of miniSEED data or limit to metadata listed
            on or before this date

        Any parameters that are omitted will be assumed to be '*' wildcards.

        >>> l1 = 'AB CDE 01 BHZ 2015-04-25T02:45:32 2015-04-25T02:47:00'
        >>> l2 = '* * * * * *'
        >>> l3 = 'AB CDE 01 BHZ 2015-04-25T00:00:00 2015-04-25T02:47:00'
        >>> l4 = 'AB CDE 01 BHZ 2015-04-25T02:45:32 2015-04-25T03:00:00'
        >>> FDSNBulkRequestItem(line=l1)
        AB CDE 01 BHZ 2015-04-25T02:45:32.000 2015-04-25T02:47:00.000
        >>> FDSNBulkRequestItem(line=l2)
        * * * * * *
        >>> A = FDSNBulkRequestItem(line=l3) #   [-------]
        >>> B = FDSNBulkRequestItem(line=l4) #        [-------]
        >>> C = FDSNBulkRequestItem(line=l1) #        [--]
        >>> D = FDSNBulkRequestItem(line=l2) # <---------------->
        >>> A.contains(l1)
        True
        >>> B.contains(C) and C.contains(C) and D.contains(C)
        True
        >>> C.contains(A) or C.contains(B) or C.contains(D)
        False

        >>> FDSNBulkRequestItem(network='IU', station='ANMO', location='  ',
        ...                     channel='BHZ', starttime='2012-04-25',
        ...                     endtime='2012-06-12T10:10:10')
        IU ANMO -- BHZ 2012-04-25T00:00:00.000 2012-06-12T10:10:10.000
        """

        def convert_time(in_time=None):
            """
            Convenience function for converting time
            """
            if in_time is None:
                return None
            elif isinstance(in_time, string_types):
                if in_time != '*':
                    return UTCDateTime(in_time)
            elif isinstance(in_time, UTCDateTime):
                return in_time
            else:
                raise NotImplementedError("cannot convert to UTCDateTime")

        # kwargs is ignored & discarded.
        if line:
            if len(line.splitlines()) > 1:
                raise ValueError(
                    "FDSNBulkRequestItem can only accept individual lines.")
            try:
                network, station, location, channel, starttime, endtime =\
                    line.split()
            except:
                print(line, file=sys.stderr)
                print("PROBLEM! count %d, but parsed into 6 items"
                      % len(line.split()))
                raise

        if location == '  ':
            location = '--'

        self.network = None if network == '*' else network
        self.station = None if station == '*' else station
        self.location = None if location == '*' else location
        self.channel = None if channel == '*' else channel

        self.starttime = convert_time(starttime)
        self.endtime = convert_time(endtime)

    def __str__(self):
        """
        String representation suitable for a bulk request, with times to ms
        """
        net = self.network or '*'
        sta = self.station or '*'
        loc = self.location or '*'
        cha = self.channel or '*'
        startt = self.starttime and self.starttime.format_iris_web_service()
        endt = self.endtime and self.endtime.format_iris_web_service()
        return (" ".join((net, sta, loc, cha, startt or '*', endt or '*')))

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        """
        Returns whether two FDSNBulkRequestItem objects match exactly

        :type other: :class:`~obspy.clients.fdsn.routers.FDSNBulkRequestItem`
        :param other: FDSNBulkRequestItem to compare against
        :rtype: bool
        :returns: true if all fields match [exactly]
        """
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        raise NotImplementedError()

    def __lt__(self, other):
        """
        less-than comparison for FDSNBulkRequestItem

        compares (in order):
            network, station, location, channel, [alphabetically]
        then
            starttime [numerically].
        Fields with wildcards are skipped

        :note: does NOT compare endtime

        :type other: :class:`~obspy.clients.fdsn.routers.FDSNBulkRequestItem`
        :param other: FDSNBulkRequestItem to compare against
        :rtype: bool
        :returns: True if self < other
        """
        if (self.network or "") != (other.network or ""):
            return (self.network or "") < (other.network or "")
        if (self.station or "") != (other.station or ""):
            return (self.station or "") < (other.station or "")
        if (self.location or "") != (other.location or ""):
            return (self.location or "") < (other.location or "")
        if (self.channel or "") != (other.channel or ""):
            return (self.channel or "") < (other.channel or "")
        if self.starttime is None and other.starttime is None:
            return False
        if self.starttime is None:
            return True
        if other.starttime is None:
            return False
        return self.starttime < other.starttime
        # do not compare endtime.

    def contains(self, other):
        """
        comparison that accounts for simple * wildcard and time

        :type other: obspy.clients.fdsn.routers.FDSNBulkRequestItem or str
        :param other: another FDSNBulkRequestItem to compare to
        :rtype: bool
        :returns: true if this item can "contain" the other.

        For example:
        >>> A = FDSNBulkRequestItem("IU ANMO * BHZ 2012-05-06 2012-05-07")
        >>> A.contains("IU ANMO 00 BHZ 2012-05-06 2012-05-07")
        True
        >>> A.contains("IU ANMO * BHZ 2012-05-06T10:00:00 2012-05-06T12:00:00")
        True
        >>> B = FDSNBulkRequestItem("IU ANMO * * 2012-05-06 2012-05-07")
        >>> A.contains(B)
        False
        >>> B.contains(A)
        True

        Note: wildcards such as B?Z or B* will not work. only solo '*' works
        """
        if isinstance(other, string_types):
            # recommend converting before comparing...
            other = FDSNBulkRequestItem(line=other)

        if not isinstance(other, self.__class__):
            raise NotImplementedError()

        return (
            (self.network is None or self.network == other.network) and
            (self.station is None or self.station == other.station) and
            (self.location is None or self.location == other.location) and
            (self.channel is None or self.channel == other.channel) and
            (self.starttime is None or self.starttime <= other.starttime) and
            (self.endtime is None or self.endtime >= other.endtime))


class FDSNBulkRequests(object):
    """
    Handles a set of :class:`~obspy.clients.fdsn.routers.FDSNBulkRequestItem`,
    allowing them to be retrieved as a big text string, and providing the
    ability to do set comparisons.

    >>> samp1 = "* * * * * *\\nAB CD -- EHZ 2015-04-23 2016-04-23T13:04:00"
    >>> text2 = "AB CD -- EHZ 2015-04-23T00:00:00 2016-04-23T13:04:00"
    >>> samp2 = FDSNBulkRequestItem(line=text2)
    >>> print(FDSNBulkRequests(samp1))
    * * * * * *
    AB CD -- EHZ 2015-04-23T00:00:00.000 2016-04-23T13:04:00.000
    >>> a = FDSNBulkRequests(samp1)
    >>> print(str(a))
    * * * * * *
    AB CD -- EHZ 2015-04-23T00:00:00.000 2016-04-23T13:04:00.000
    >>> a.add(text2)
    >>> print(str(a))
    * * * * * *
    AB CD -- EHZ 2015-04-23T00:00:00.000 2016-04-23T13:04:00.000
    """

    def __init__(self, items):
        """
        Initializer for FDSNBulkRequests

        :type items: str or collection of FDSNBulkRequestItem
        :param items: string may be newline-separated bulk request text block
        """
        if not items:
            self.items = set()
        elif isinstance(items, string_types):
            self.items = {FDSNBulkRequestItem(line=item)
                          for item in items.splitlines()}
        elif isinstance(items, collections.Iterable):
            self.items = {item if isinstance(item, FDSNBulkRequestItem)
                          else FDSNBulkRequestItem(item) for item in items}
        else:
            self.items = set()

    def __iter__(self):
        return self.items.__iter__()

    def __str__(self):
        """
        get a sorted string representation of the bulk request
        """
        ordered_items = sorted(self.items)
        return "\n".join([str(item) for item in ordered_items])

    def __len__(self):
        return len(self.items)

    def add(self, val):
        """
        add an FDSNBulkRequestItem to this FDSNBulkRequests object

        :type val: :class:`~obspy.clients.fdsn.routers.FDSNBulkRequestItem`
        :param val: item to be added. If not FDSNBulkRequestsItem, an attempt
        will be made to convert it
        """
        if not isinstance(val, FDSNBulkRequestItem):
            self.items.add(FDSNBulkRequestItem(line=val))
        else:
            self.items.add(val)

    def update(self, val):
        """
        update with the union between it and another FDSNBulkRequests object

        :type val: :class:`~obspy.clients.fdsn.routers.FDSNBulkRequests` or
        something convertable to FDSNBulkRequests
        :param val: set of requests that will be merged into this
        """
        if not val:
            return
        if isinstance(val, FDSNBulkRequests):
            self.items.update(val.items)
        else:
            self.items.update((FDSNBulkRequests(val)))

    def difference_update(self, val):
        """
        remove all requests that are found in another FDSNBulkRequests object

        :type val: :class:`~obspy.clients.fdsn.routers.FDSNBulkRequests` or
        something convertable to FDSNBulkRequests
        :param val: set of requests that will be removed from this
        """
        if not val:
            return
        if isinstance(val, FDSNBulkRequests):
            self.items.difference_update(val.items)
        else:
            self.items.difference_update((FDSNBulkRequests(val)))


def inventory_to_bulkrequests(inv):
    """
    convert from Inventory to FDSNBulkRequests

    :type inv: `~obspy.core.inventory.inventory.Inventory`
    :param inv: ObsPy Stream data (aka, station metadata)
    :rtype: `~obspy.clients.fdsn.routers.FDSNBulkRequests`
    :returns: flat representation of inventory tree with duplicates removed
    """
    bulk = FDSNBulkRequests(None)
    for network in inv.networks:
        net = network.code
        if not network.stations:
            bulk.add(FDSNBulkRequestItem(network=net,
                                         starttime=network.start_date,
                                         endtime=network.end_date))
            continue
        for station in network.stations:
            sta = station.code
            if not station.channels:
                bulk.add(FDSNBulkRequestItem(network=net, station=sta,
                                             starttime=station.start_date,
                                             endtime=station.end_date))
                continue
            for channel in station.channels:
                loc = channel.location_code or '  '
                cha = channel.code
                bulk.add(FDSNBulkRequestItem(network=net, station=sta,
                                             location=loc, channel=cha,
                                             starttime=channel.start_date,
                                             endtime=channel.end_date))
    return bulk


def stream_to_bulkrequests(data):
    """
    convert waveforms to FDSNBulkRequests

    :type data: `~obspy.core.stream.Stream`
    :param data: obspy Stream data (aka, waveforms)
    :rtype: `~obspy.clients.fdsn.routers.FDSNBulkRequests`
    :returns: flat representation of inventory tree with duplicates removed
    """
    return FDSNBulkRequests({FDSNBulkRequestItem(**d.stats) for d in data})


class RoutingResponse(object):
    """
    Serves as an abstract base for routes.  A route is the combination of a
    provider along with multiple specific requests.

    """

    def __init__(self, provider_id, raw_requests=None):
        """
        initializer for RoutingRespones

        :type provider_id: str
        :param provider_id: provider_id for the data provider
        :type raw_requests: iterable
        :param raw_requests: requests to be interpreted and passed to provider
        """
        self.provider_id = provider_id
        self.request_items = raw_requests

    def __len__(self):
        if self.request_items:
            return len(self.request_items)
        return 0

    def __str__(self):
        if len(self) != 1:
            item_or_items = " items"
        else:
            item_or_items = " item"
        return self.provider_id + ", with " + str(len(self)) + item_or_items

    def add_request(self, line):
        """
        ABSTRACT add a request to the RoutingResponse

        :type line: variable, defined by each subclass
        :param line: describes item to add to response
        """
        raise NotImplementedError("RoutingResponse.add_request()")


class FederatedRoute(RoutingResponse):
    """ #pylint ignore
    Route devoted to an FDSN provider.  It contains the who, what, and where
    needed to send requests to a service, and provides the facilities for
    adding service endpoints, request items, and additional query parameters.
    Requests are stored as FDSNBulkRequests

    >>> fed_resp = FederatedRoute("IRISDMC")
    >>> fed_resp.add_query_param(["lat=50","lon=20","level=cha"])
    >>> svc_url = "http://service.iris.edu/fdsnws/station/1/"
    >>> fed_resp.add_service("STATIONSERVICE", svc_url)
    >>> fed_resp.add_request("AI ORCD -- BHZ 2015-01-01T00:00:00 "
    ...                      "2016-01-02T00:00:00")
    >>> fed_resp.add_request("AI ORCD 04 BHZ 2015-01-01T00:00:00 "
    ...                      "2016-01-02T00:00:00")
    >>> print(fed_resp.text("STATIONSERVICE"))
    level=cha
    AI ORCD -- BHZ 2015-01-01T00:00:00.000 2016-01-02T00:00:00.000
    AI ORCD 04 BHZ 2015-01-01T00:00:00.000 2016-01-02T00:00:00.000
    """

    pass_through_params = {
        "DATASELECTSERVICE": ["longestonly", "quality", "minimumlength"],
        "STATIONSERVICE": [
            "level", "matchtimeseries", "includeavailability",
            "includerestricted", "format"
        ]
    }

    def __init__(self, provider_id):
        """
        initialize a FederatedRoute

        :type provider_id: str
        :param provider_id: short code used to identify the provider

        :note: The codes provided by the Fedcatalog do not necessarily match
        those used by ObsPy
        """
        RoutingResponse.__init__(self, provider_id,
                                 raw_requests=FDSNBulkRequests(None))
        self.parameters = []
        self.services = {}

    def add_service(self, service_name, service_url):
        """
        add a service url to this response

        :type service_name: str
        :param service_name: name such as STATIONSERVICE, DATASELECTSERVICE,
        or DATACENTER
        :param service_url: str
        :param service_url: url of service, like
        http://service.iris.edu/fdsnws/station/1/
        """
        self.services[service_name] = service_url

    def add_query_param(self, parameters):
        """
        add query parameters to list.  These may be prepended to a bulk request

        :type parameters: str or FedcatResponseLine or iterable of str
        :param parameters: strings of the form "param=value"

        >>> fedresp = FederatedRoute("ABC")
        >>> fedresp.add_query_param(["level=station","quality=D"])
        >>> fedresp.add_query_param("onceuponatime=now")
        >>> fedresp.add_query_param(FedcatResponseLine("testing=true"))
        >>> fedresp.add_query_param([FedcatResponseLine("black=white"),
        ...                          FedcatResponseLine("this=that")])
        >>> print(",".join(fedresp.parameters))
        level=station,quality=D,onceuponatime=now,testing=true,black=white,this=that
        """
        if isinstance(parameters, string_types):
            self.parameters.append(parameters)
        elif isinstance(parameters, FedcatResponseLine):
            self.parameters.append(str(parameters))
        else:
            self.parameters.extend([str(p) for p in parameters])

    def add_request(self, lines):
        """append request(s) to the list of requests

        :type lines:
        :param lines: string or FedcatResponseLine that looks something like:
        NET STA LOC CHA yyyy-mm-ddTHH:MM:SS yyyy-mm-ddTHH:MM:SS
        """
        if isinstance(lines, (string_types, FedcatResponseLine)):
            self.request_items.update(str(lines))  # no returns expected!
        elif isinstance(lines, collections.Iterable):
            self.request_items.add([str(line) for line in lines])

    def text(self, target_service):
        """
        Return a string suitable for posting to a target service

        :type target_service: str
        :param target_service: name of target service, like 'DATASELECTSERVICE'
        :rtype: str
        :returns: bulk request text
        """
        reply = []
        for good in FederatedRoute.pass_through_params[target_service]:
            reply.extend(
                [c for c in self.parameters if c.startswith(good + "=")])
        if reply:
            params_str = "\n".join(reply)
            return "\n".join((params_str, str(self.request_items)))
        else:
            return str(self.request_items)

    def get_service_mappings(self):
        """
        return service mappings for the federated route

        :rtype: dict
        :return: the service mapping for the given federated route

        >>> fed_resp = FederatedRoute("IRISDMC")
        >>> svc_url = "http://service.iris.edu/fdsnws/station/1/"
        >>> fed_resp.add_service("STATIONSERVICE", svc_url)
        >>> fed_resp.get_service_mappings()
        {'station': 'http://service.iris.edu/fdsnws/station/1'}
        >>> fed_resp = FederatedRoute("IRISDMC")
        >>> svc_url = "http://service.iris.edu/fdsnws/dataselect/1/"
        >>> fed_resp.add_service("DATASELECTSERVICE", svc_url)
        >>> fed_resp.get_service_mappings()
        {'dataselect': 'http://service.iris.edu/fdsnws/dataselect/1'}
        """
        service_mappings = {}
        if self.services.get('STATIONSERVICE'):
            service_mappings['station'] = self.services \
                                        .get('STATIONSERVICE') \
                                        .strip('/')
        if self.services.get('DATASELECTSERVICE'):
            service_mappings['dataselect'] = self.services \
                                        .get('DATASELECTSERVICE') \
                                        .strip('/')
        return service_mappings

    def __str__(self):
        """
        return a summary representation of this item
        """
        out = "FederatedRoute for {id} containing {pcount} "\
              "query parameters and {rcount} request items".\
              format(id=self.provider_id, pcount=len(self.parameters),
                     rcount=len(self.request_items))
        return out


class FedcatResponseLine(object):
    """
    line from federated catalog source that provides additional tests

    >>> fed_text = '''minlat=34.0
    ...
    ... DATACENTER=GEOFON,http://geofon.gfz-potsdam.de
    ... DATASELECTSERVICE=http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/
    ... CK ASHT -- HHZ 2015-01-01T00:00:00 2016-01-02T00:00:00
    ...
    ... DATACENTER=INGV,http://www.ingv.it
    ... STATIONSERVICE=http://webservices.rm.ingv.it/fdsnws/station/1/
    ... HL ARG -- BHZ 2015-01-01T00:00:00 2016-01-02T00:00:00
    ... HL ARG -- VHZ 2015-01-01T00:00:00 2016-01-02T00:00:00'''

    >>> for y in fed_text.splitlines():
    ...    x = FedcatResponseLine(y)
    ...    edprs = [x.is_empty(), x.is_datacenter(), x.is_param(),
    ...             x.is_request(), x.is_service()]
    ...    print("\\n".join([str(edprs)]))
    [False, False, True, False, False]
    [True, False, False, False, False]
    [False, True, True, False, False]
    [False, False, True, False, True]
    [False, False, False, True, False]
    [True, False, False, False, False]
    [False, True, True, False, False]
    [False, False, True, False, True]
    [False, False, False, True, False]
    [False, False, False, True, False]
    """

    def is_empty(self):
        """
        :rtype: bool
        :returns: true if self is empty

        >>> line = '\\n'
        >>> fcr = FedcatResponseLine(line)
        >>> fcr.is_empty()
        True
        """
        return self.line == ""

    def is_datacenter(self):
        """
        :rtype: bool
        :returns: true if self contains datacenter details

        >>> line = 'DATACENTER=IRIS-DMC,https://...'
        >>> fcr = FedcatResponseLine(line)
        >>> fcr.is_datacenter()
        True
        """
        return self.line.startswith('DATACENTER=')

    def is_param(self):
        """
        :rtype: bool
        :returns: true if self could be a param=value

        >>> line = 'level=station'
        >>> fcr = FedcatResponseLine(line)
        >>> fcr.is_param()
        True
        """
        # true for provider, services, and parameter_list
        return '=' in self.line

    def is_request(self):
        """
        :rtype: bool
        :returns: true if self might be interpreted as a request line

        >>> line = 'CK ASHT -- HHZ 2015-01-01T00:00:00 2016-01-02T00:00:00'
        >>> fcr = FedcatResponseLine(line)
        >>> fcr.is_request()
        True
        """
        return len(self.line.split()) == 6  # and test field values?

    def is_service(self):
        """
        :rtype: bool
        :returns: true if a parameter that might be pointing to a service

        >>> line = 'DATASELECTSERVICE=https://...'
        >>> fcr = FedcatResponseLine(line)
        >>> fcr.is_param()
        True
        """
        return self.is_param() and self.line.split(
            "=")[0].isupper() and not self.is_datacenter()

    def __init__(self, line):
        self.line = line.strip()

    def __repr__(self):
        return self.line

    def __str__(self):
        return self.line


class ParserState(object):
    """
    Parsers leverage the known structure of Fedcatalog's response

    PREPARSE -> [PARAMLIST | EMPTY_LINE | DATACENTER]
    PARAMLIST -> [PARAMLIST | EMPTY_LINE]
    EMPTY_LINE -> [EMPTY_LINE | DATACENTER | DONE]
    DATACENTER -> [SERVICE]
    SERVICE -> [SERVICE | REQUEST]
    REQUEST -> [REQUEST | EMPTY_LINE | DONE ]
    """

    @staticmethod
    def parse(line, this_response):
        """abstract"""
        raise NotImplementedError("ParserState.parse()")

    @staticmethod
    def next(line):
        """abstract"""
        raise NotImplementedError("ParserState.next()")


class PreParse(ParserState):
    """Initial ParserState for federated response"""

    @staticmethod
    def parse(line, this_response):
        return this_response

    @staticmethod
    def next(line):
        if line.is_empty():
            return EmptyItem
        elif line.is_datacenter():
            return DatacenterItem
        elif line.is_param():
            return ParameterItem
        else:
            return ParserState


class ParameterItem(ParserState):
    """handle a parameter from federated response"""

    @staticmethod
    def parse(line, this_response):
        """Parse: param=value"""
        this_response.add_query_param(line)
        return this_response

    @staticmethod
    def next(line):
        if line.is_empty():
            return EmptyItem
        elif line.is_param():
            return ParameterItem
        else:
            raise RuntimeError(
                "Expect parameter to be followed by another, or an empty line"
            )


class EmptyItem(ParserState):
    """handle an empty line from federated response"""

    @staticmethod
    def parse(line, this_response):
        return this_response

    @staticmethod
    def next(line):
        if line.is_empty():
            return EmptyItem
        elif line.is_datacenter():
            return DatacenterItem
        else:
            raise RuntimeError(
                "expected either a DATACENTER or another empty line [" +
                str(line) + "]")


class DatacenterItem(ParserState):
    """handle data center from federated response"""

    @staticmethod
    def parse(line, this_response):
        """Parse: DATACENTER=id,http://url..."""
        _, rest = str(line).split('=')
        active_id, url = rest.split(',')
        this_response = FederatedRoute(active_id)
        this_response.add_service("DATACENTER", url)
        return this_response

    @staticmethod
    def next(line):
        if line.is_service():
            return ServiceItem
        else:
            raise RuntimeError(
                "DATACENTER line should be followed by a service")


class ServiceItem(ParserState):
    """handle service description from federated response"""

    @staticmethod
    def parse(line, this_response):
        """Parse: SERVICENAME=http://service.url/"""
        svc_name, url = str(line).split('=')
        this_response.add_service(svc_name, url)
        return this_response

    @staticmethod
    def next(line):
        if line.is_service():
            return ServiceItem
        elif line.is_request():
            return RequestItem
        else:
            raise RuntimeError(
                "Expeted Service to be followed by request or more services"
            )


class RequestItem(ParserState):
    """handle request lines from federated response"""

    @staticmethod
    def parse(line, this_response):
        """Parse: NT STA LC CHA YYYY-MM-DDThh:mm:ss YY-MM-DDThh:mm:ss"""
        this_response.add_request(line)
        return this_response

    @staticmethod
    def next(line):
        if line.is_request():
            return RequestItem
        elif line.is_empty():
            return EmptyItem
        else:
            raise RuntimeError(
                "Requests should be followed by another request or empty line"
            )


# main function
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
