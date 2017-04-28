#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FDSN Web service client for ObsPy.

:copyright:
    ?
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import print_function
# import sys
import collections
# from collections import OrderedDict
from threading import Lock
#import warnings
import os
import requests
from future.utils import native_str
# from requests.exceptions import (HTTPError, Timeout)
#from obspy.core import UTCDateTime
from obspy.core.inventory import Inventory
from obspy.core import Stream
from obspy.clients.fdsn.client import convert_to_string
from obspy.clients.fdsn.header import FDSNException
from obspy.clients.fdsn.routers.routing_client import (RoutingClient,
                                                       RoutingManager, logger)
from obspy.clients.fdsn.routers import (FederatedRoute,)
from obspy.clients.fdsn.routers.fedcatalog_parser import (PreParse,
                                                          FedcatResponseLine,
                                                          DatacenterItem,
                                                          inventory_to_bulkrequests,
                                                          stream_to_bulkrequests)


# IRIS uses different codes for datacenters than obspy.
#         (iris_name , obspy_name)
REMAPS = (("IRISDMC", "IRIS"),
          ("GEOFON", "GFZ"),
          ("SED", "ETH"),
          ("USPC", "USP"))

FEDCATALOG_URL = 'https://service.iris.edu/irisws/fedcatalog/1/'

def inv2set(inv, level):
    """
    used to quickly decide what exists and what doesn't

    :type inv:
    :param inv:
    :type level:
    :param level:
    :rtype:
    :return:
    """

    converter = {
        "channel": channel_set,
        "response": channel_set,
        "station": station_set,
        "network": network_set
    }

    def channel_set(inv):
        """
        return a set containing string representations of an inv object
        """
        return {
            n.code + "." + s.code + "." + c.location_code + "." + c.code
            for n in inv for s in n for c in s
        }

    def station_set(inv):
        """
        return a set containing string representations of an inv object
        """
        return {n.code + "." + s.code for n in inv for s in n}

    def network_set(inv):
        """
        return a set containing string representations of an inv object
        """
        return {n.code for n in inv}

    return converter[level](inv)


# converters used to make comparisons between inventory items and requests

def distribute_args(argdict):
    """
    divide a dictionary's keys between fedcatalog and fdsnservice

    :type argdict: dict
    :param argdict:
    :rtype:
    :return: tuple of dictionaries fedcat_kwargs, fdsn_kwargs
    """
    # TODO figure out where each of the arguments belongs
    #  to the fedrequest?  to the final service?

    fedcatalog_params = ('')
    fedcatalog_prohibited_params = ('filename', 'attach_response', 'user', 'password')
    service_params = ('user', 'password', 'attach_response')

    # fedrequest gets almost all arguments, except for some
    fed_argdict = argdict.copy()
    for key in fedcatalog_prohibited_params:
        if key in fed_argdict:
            del fed_argdict[key]

    # services get practically no arguments, since they're provided by the bulk request
    service_args = dict()
    for key in service_params:
        if key in argdict:
            service_args[key] = argdict[key]
    return fed_argdict, service_args


# the following is ripped out of fdsn.client.Client because it doesn't need to be
# be associated with the client class. TODO: pull out of client class, and then
# just import it
def _get_bulk_string(bulk, arguments):
    # If its an iterable, we build up the query string from it
    # StringIO objects also have __iter__ so check for 'read' as well

    args = ["%s=%s" % (key, convert_to_string(value))
            for key, value in arguments.items() if value is not None]

    if isinstance(bulk, collections.Iterable) \
            and not hasattr(bulk, "read") \
            and not isinstance(bulk, (str, native_str)):
        # empty location codes have to be represented by two dashes
        tmp = [" ".join((net, sta, loc or "--", cha,
                         convert_to_string(t1), convert_to_string(t2)))
               for net, sta, loc, cha, t1, t2 in bulk]
        tmp = "\n".join(tmp)
    else:
        # if it has a read method, read data from there
        if hasattr(bulk, "read"):
            tmp = bulk.read()
        elif isinstance(bulk, (str, native_str)):
            # check if bulk is a local file
            if "\n" not in bulk and os.path.isfile(bulk):
                with open(bulk, 'r') as fh:
                    tmp = fh.read()
            # just use bulk as input data
            else:
                tmp = bulk
        else:
            msg = ("Unrecognized input for 'bulk' argument. Please "
                   "contact developers if you think this is a bug.")
            raise NotImplementedError(msg)
    if args:
        args = '\n'.join(args)
        bulk = '\n'.join((args, tmp))
    else:
        bulk = tmp
    logger.info(bulk)
    assert isinstance(bulk, (str, native_str))
    return bulk

class FedcatalogProviders(object):
    """
    Class containing datacenter details retrieved from the fedcatalog service

    keys: name, website, lastupdate, serviceURLs {servicename:url,...},
    location, description

    >>> prov = FedcatalogProviders()
    >>> print(prov.pretty('IRISDMC'))  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    IRISDMC:The IRIS Data Management Center, Seattle, WA, USA WEB:http://ds.iris.edu  LastUpdate:...M

    """

    def __init__(self):
        self._providers = dict()
        self._lock = Lock()
        self._failed_refreshes = 0
        self.refresh()

    def __iter__(self):
        """
        iterate through each provider name
        >>> fcp=FedcatalogProviders()
        >>> print(sorted([fcp.get(k,'name') for k in fcp]))  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ['BGR',..., 'USPSC']

        """
        if not self._providers:
            self.refresh()
        return self._providers.__iter__()

    @property
    def names(self):
        """
        get names of datacenters

        >>> fcp=FedcatalogProviders()
        >>> print(sorted(fcp.names))  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ['BGR',..., 'USPSC']
        """
        if not self._providers:
            self.refresh()
        return self._providers.keys()

    def get(self, name, detail=None):
        """
        get a datacenter property

        >>> fcp = FedcatalogProviders()
        >>> fcp.get('ORFEUS','description')
        'The ORFEUS Data Center'

        :type name: str
        :param name: provider name. such as IRISDMC, ORFEUS, etc.
        :type detail: str
        :param detail: property of interest.  eg, one of ('name', 'website',
        'lastupdate', 'serviceURLs', 'location', 'description').
        if no detail is provided, then the entire dict for the requested provider
        will be returned
        """
        if not self._providers:
            self.refresh()
        if not name in self._providers:
            return ""
        else:
            if detail:
                return self._providers[name][detail]
            else:
                return self._providers[name]

    def refresh(self, force=False):
        """
        retrieve provider profile from fedcatalog service

        >>> providers = FedcatalogProviders()
        >>> # providers.refresh(force=True)
        >>> providers.names #doctest: +ELLIPSIS
        dict_keys(['...'])

        :type force: bool
        :param force: attempt to retrieve data even if too many failed attempts
        """
        if  self._providers and not force:
            return
        if self._lock.locked():
            return
        with self._lock:
            logger.debug("Refreshing Provider List")
            if self._failed_refreshes > 3 and not force:
                logger.error(
                    "Unable to retrieve provider profiles from fedcatalog service after {0} attempts"
                    % (self._failed_refreshes))

            try:
                url = 'https://service.iris.edu/irisws/fedcatalog/1/datacenters'
                r = requests.get(url, verify=False)
                self._providers = {v['name']: v for v in r.json()}
                self._failed_refreshes = 0
            except:
                logger.error(
                    "Unable to update provider profiles from fedcatalog service")
                self._failed_refreshes += 1
            else:
                for iris_name, obspy_name in REMAPS:
                    if iris_name in self._providers:
                        self._providers[obspy_name] = self._providers[iris_name]

    def pretty(self, name):
        """
        return nice text representation of service without too much details
        >>> providers = FedcatalogProviders()
        >>> print(providers.pretty("ORFEUS"))  #doctest: +ELLIPSIS
        ORFEUS:The ORFEUS Data Center, de Bilt, the Netherlands WEB:http://www.orfeus-eu.org  LastUpdate:...M
        >>> print(providers.pretty("IRIS") == providers.pretty("IRISDMC"))
        True
        """
        if not self._providers:
            self.refresh()
        if not name in self._providers:
            return ""
        return "{name}:{description}, {location} WEB:{website}  LastUpdate:{lastUpdate}".format(**self._providers[name])
        #fields = ("name", "description", "location", "website", "lastUpdate")
        #return '\n'.join(self._providers[name][k] for k in fields) + '\n'


PROVIDERS = FedcatalogProviders()

class FederatedClient(RoutingClient):
    """
    FDSN Web service request client.

    For details see the :meth:`~obspy.clients.fdsn.client.Client.__init__()`
    method.
    >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
    >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    >>> client = FederatedClient()
    >>> print(client)  #doctest: +ELLIPSIS
    Federated Catalog Routing Client

    >>> inv = client.get_stations(network="I?", station="AN*", channel="*HZ")
    ...                           #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    >>> print(inv)  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    Inventory created at ...Z
    	Created by: IRIS WEB SERVICE: fdsnws-station | version: 1...
    		    http://service.iris.edu/fdsnws/station/1/query
    	Sending institution: IRIS-DMC (IRIS-DMC)
    	Contains:
    		Networks (1):
    			IU
    		Stations (10):
    			IU.ANMO (Albuquerque, New Mexico, USA) (6x)
    			IU.ANTO (Ankara, Turkey) (4x)
    		Channels (0):
    <BLANKLINE>

    >>> inv = client.get_stations(network="I?", station="AN*", channel="*HZ", filename=sys.stderr)
    ...                           #doctest: +SKIP
    .. Warning: if output is sent directly to a file, then the success
                status will not be checked beyond gross failures (no data, no response, timeout)
    """
    def __init__(self, **kwargs):
        """
        """
        RoutingClient.__init__(self, **kwargs)
        PROVIDERS.refresh()
    def __str__(self):
        ret = "Federated Catalog Routing Client"
        return ret

    def get_routing(self, routing_file=None, **kwargs):
        """
        send query to the fedcatalog service

        >>> client = FederatedClient()
        >>> params={"station":"ANTO","includeoverlaps":"true"}
        >>> frm = client.get_routing(**params)
        >>> for f in frm:
        ...   print(f)  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        FederatedRoute for IRIS containing 0 query parameters and ... request items
        FederatedRoute for ORFEUS containing 0 query parameters and ... request items

        :type params:
        :param params:
        :rtype: :class:`~obspy.clients.fdsn.routers.FederatedRoutingManager`
        :return: parsed response from the FedCatalog service
        """
        assert not 'bulk' in kwargs
        resp = requests.get(FEDCATALOG_URL + "query", params=kwargs, verify=False)
        resp.raise_for_status()
        if routing_file:
            pass #write out to file

        frm = FederatedRoutingManager(resp.text)
        return frm

    def get_routing_bulk(self, bulk=None, routing_file=None, **kwargs):
        """
        send query to the fedcatalog service as a POST.

        >>> client = FederatedClient()
        >>> params={"includeoverlaps":"true"}
        >>> frm = client.get_routing_bulk(bulk="* ANTO * * * *", **params)
        >>> for f in frm:
        ...   print(f)  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        FederatedRoute for IRIS containing 0 query parameters and ... request items
        FederatedRoute for ORFEUS containing 0 query parameters and ... request items

        :type bulk:
        :param bulk:
        :rtype: :class:`~obspy.clients.fdsn.routers.FederatedRoutingManager`
        :return: parsed response from the FedCatalog service
        """
        if isinstance(bulk, collections.Iterable):
             bulk = _get_bulk_string(bulk, kwargs)
        resp = requests.post(FEDCATALOG_URL + "query", data=bulk, verify=False)
        resp.raise_for_status()
        logger.info(resp.text)
        frm = FederatedRoutingManager(resp.text)
        return frm

    def _request(self, client=None, service=None, route=None, output=None, passed=None, failed=None, filename=None, **kwargs):
        """
        function used to query FDSN webservice using

        :meth:`~obspy.clients.fdsn.client.Client.get_waveforms_bulk` or
        :meth:`~obspy.clients.fdsn.client.Client.get_stations_bulk`

        :type client:
        :param client:
        :type service:
        :param service:
        :type route: :class:`~obspy.clients.fdsn.route.FederatedRoute`
        :param route:
        :type output: container accepting "put"
        :param output: place where retrieved data go
        :type failed: contdainer accepting "put"
        :param failed: place where list of unretrieved bulk request lines go
        :param filename:
        :type **kwargs:
        :param **kwargs:
        """
        assert isinstance is not None
        assert service in ("DATASELECTSERVICE", "STATIONSERVICE"), "couldn't find {0}\n".format(service)
        assert route is not None
        assert output is not None
        assert failed is not None

        bulk_services = {"DATASELECTSERVICE": client.get_waveforms_bulk,
                         "STATIONSERVICE": client.get_stations_bulk}
        try:
            get_bulk = bulk_services.get(service)
        except ValueError:
            valid_services = '"' + ', '.join(bulk_services.keys)
            raise ValueError("Expected one of " + valid_services + " but got {0}",
                             service)

        try:
            if isinstance(filename, str):
                filename = route.provider_id + "-" + filename
            if filename:
                get_bulk(bulk=route.text(service), filename=filename, **kwargs)
            else:
                data = get_bulk(bulk=route.text(service), filename=filename, **kwargs)
                passed = data_to_request(data)
                logger.info("Retrieved {0} items from {1}".format(len(passed), route.provider_id))
                logger.info('\n'+ str(passed))
                output.put(data)
        except FDSNException as ex:
            failed.put(route.request_items)
            print("Failed to retrieve data from: {0}", route.provider_id)
            print(ex)
            raise

    def get_waveforms_bulk(self,
                           bulk,
                           quality=None,
                           minimumlength=None,
                           longestonly=None,
                           filename=None,
                           includeoverlaps=False,
                           reroute=False,
                           **kwargs):
        """
        retrieve waveforms from data providers via POST request to the Fedcatalog service

        :type includeoverlaps: boolean
        :param includeoverlaps: retrieve same information from multiple sources
        (not recommended)
        :type reroute: boolean
        :param reroute: if data doesn't arrive from provider , see if it is available elsewhere
        other parameters as seen in :meth:`~obspy.fdsn.clients.Client.get_waveforms_bulk`
        and :meth:`~obspy.fdsn.clients.Client.get_stations_bulk`
        :rtype: :class:`~obspy.core.stream.Stream`
        :return:


        >>> client = FederatedClient()
        >>> bulkreq = "IU ANMO * ?HZ 2010-02-27T06:30:00 2010-02-27T06:33:00"
        >>> tr = client.get_waveforms_bulk(bulk=bulkreq)
        ...        #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        >>> print(tr)  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        6 Trace(s) in Stream:
        IU.ANMO.00.BHZ | 2010-02-27T06:30... | 20.0 Hz, 3600 samples
        IU.ANMO.00.LHZ | 2010-02-27T06:30... | 1.0 Hz, 180 samples
        ...
        IU.ANMO.10.VHZ | 2010-02-27T06:30... | 0.1 Hz, 18 samples

        """

        fed_kwargs, svc_kwargs = distribute_args(kwargs)
        fed_kwargs["includeoverlaps"] = includeoverlaps

        # bulk = _get_bulk_string(bulk, fed_kwargs)
        frm = self.get_routing_bulk(bulk) #, fed_kwargs)
        data, passed, failed  = self.query(frm, "DATASELECTSERVICE", **svc_kwargs)

        if reroute and failed:
            logger.info(str(len(failed)) + " items were not retrieved, trying again," +
                        " but from any provider (while still honoring include/exclude)")
            fed_kwargs["includeoverlaps"] = True
            frm = self.get_routing_bulk(bulk=failed.text(), **fed_kwargs)
            more_data, passed, failed = self.query(frm, "DATASELECTSERVICE", keep_unique=True, **svc_kwargs)
            logger.info("Retrieved {0} additional items".format(len(passed)))
            logger.info("Unable to retrieve {0} items:".format(len(failed)))
            logger.info(str(failed))
            data += more_data

        return data

    def get_waveforms(self, network, station, location, channel, starttime, endtime,
                      includeoverlaps=False, reroute=False, **kwargs):
        """
        retrieve waveforms from data providers via GET request to the Fedcatalog service

        :type includeoverlaps: boolean
        :param includeoverlaps: retrieve same information from multiple sources
        (not recommended)
        :type reroute: boolean
        :param reroute: if data doesn't arrive from provider , see if it is available elsewhere
        other parameters as seen in :meth:`~obspy.fdsn.clients.Client.get_waveforms`
        :rtype: :class:`~obspy.core.stream.Stream`
        :return:

        >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
        >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        >>> client = FederatedClient()
        >>> from obspy.core import  UTCDateTime
        >>> t_st = UTCDateTime("2010-02-27T06:30:00")
        >>> t_ed = UTCDateTime("2010-02-27T06:33:00")
        >>> tr = client.get_waveforms('IU', 'ANMO', '*', 'BHZ', t_st, t_ed)
        ...                           #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        >>> print(tr)  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        2 Trace(s) in Stream:
        IU.ANMO.00.BHZ | 2010-02-27T06:30:00... 20.0 Hz, 3600 samples
        IU.ANMO.10.BHZ | 2010-02-27T06:30:00... 40.0 Hz, 7200 samples
        """

        fed_kwargs, svc_kwargs = distribute_args(kwargs)
        fed_kwargs["includeoverlaps"] = includeoverlaps
        frm = self.get_routing(network=network, station=station,
                               location=location, channel=channel,
                               starttime=starttime, endtime=endtime, **fed_kwargs)

        data, passed, failed  = self.query(frm, "DATASELECTSERVICE", **svc_kwargs)

        if reroute and failed:
            logger.info(str(len(failed)) + " items were not retrieved, trying again," +
                        " but from any provider (while still honoring include/exclude)")
            fed_kwargs["includeoverlaps"] = True
            frm = self.get_routing_bulk(bulk=failed.text(), **fed_kwargs)
            more_data, passed, failed = self.query(frm, "DATASELECTSERVICE", keep_unique=True, **svc_kwargs)
            logger.info("Retrieved {0} additional items".format(len(passed)))
            logger.info("Unable to retrieve {0} items:".format(len(failed)))
            logger.info('\n'+ str(failed))
            data += more_data

        return data

    def get_stations_bulk(self, bulk, includeoverlaps=False, reroute=False,
                          **kwargs):
        """
        retrieve station metadata from data providers via POST request to the Fedcatalog service

        :type includeoverlaps: boolean
        :param includeoverlaps: retrieve same information from multiple sources
        (not recommended)
        :type reroute: boolean
        :param reroute: if data doesn't arrive from provider , see if it is available elsewhere
        :rtype: :class:`~obspy.core.inventory.inventory.Inventory`
        :return:
        other parameters as seen in :meth:`~obspy.fdsn.clients.Client.get_stations_bulk`

        >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
        >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        >>> client = FederatedClient()
        >>> bulktxt = "level=channel\\nA? OKS? * ?HZ * *"
        >>> INV = client.get_stations_bulk(bulktxt)  #doctest: +ELLIPSIS
        >>> print(INV)  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Inventory created at 2...Z
            Created by: IRIS WEB SERVICE: fdsnws-station | version: 1...
                    http://service.iris.edu/fdsnws/station/1/query
            Sending institution: IRIS-DMC (IRIS-DMC)
            Contains:
                Networks (1):
                    AV
                Stations (2):
                    AV.OKSO (South, Okmok Caldera, Alaska)
                    AV.OKSP (Steeple Point, Okmok Caldera, Alaska)
                Channels (5):
                    AV.OKSO..BHZ, AV.OKSP..EHZ (4x)
        """


        fed_kwargs, svc_kwargs = distribute_args(kwargs)
        fed_kwargs["includeoverlaps"] = includeoverlaps

        #bulk = _get_bulk_string(bulk, fed_kwargs)
        frm = self.get_routing_bulk(bulk) #, fed_kwargs)
        inv, passed, failed  = self.query(frm, "STATIONSERVICE", **svc_kwargs)

        if reroute and failed:
            logger.info(str(len(failed)) + " items were not retrieved, trying again," +
                        " but from any provider (while still honoring include/exclude)")
            fed_kwargs["includeoverlaps"] = True
            frm = self.get_routing_bulk(bulk=failed.text(), **fed_kwargs)
            more_inv, passed, failed = self.query(frm, "STATIONSERVICE", keep_unique=True, **svc_kwargs)
            logger.info("Retrieved {0} additional items".format(len(passed)))
            logger.info("Unable to retrieve {0} items:".format(len(failed)))
            logger.info('\n'+ str(failed))
            inv += more_inv

        return inv

    def get_stations(self,
                     includeoverlaps=False, reroute=False,
                     **kwargs):
        """
        retrieve station metadata from data providers via GET request to the Fedcatalog service

        :type includeoverlaps: boolean
        :param includeoverlaps: retrieve same information from multiple sources
        (not recommended)
        :type reroute: boolean
        :param reroute: if data doesn't arrive from provider , see if it is available elsewhere
        other parameters as seen in :meth:`~obspy.fdsn.clients.Client.get_stations`
        :rtype: :class:`~obspy.core.inventory.inventory.Inventory`
        :return:

        >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
        >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        >>> fclient = FederatedClient()
        >>> INV = fclient.get_stations(network="A?", station="OK*",
        ...                           channel="?HZ", level="station",
        ...                           endtime="2016-12-31")  #doctest: +ELLIPSIS
        >>> print(INV)  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Inventory created at 2...Z
            Created by: IRIS WEB SERVICE: fdsnws-station | version: 1...
                    http://service.iris.edu/fdsnws/station/1/query
            Sending institution: IRIS-DMC (IRIS-DMC)
            Contains:
                Networks (1):
                    AV
                Stations (14):
                    AV.OKAK (Cape Aslik 2, Okmok Caldera, Alaska)
                    ...
                    AV.OKWR (West Rim, Okmok Caldera, Alaska)
                Channels (0):
        <BLANKLINE>

        Exclude a provider from being queried

        >>> keep_out = ["IRISDMC","IRIS","IRIS-DMC"]
        >>> fclient.exclude_provider = keep_out
        >>> INV2 = fclient.get_stations(network="I?", station="A*",
        ...                           level="network")
        ...                           #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        >>> print(INV2)  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Inventory created at ...Z
            Created by: ObsPy ...
                    https://www.obspy.org
            Sending institution: SeisComP3,SeisNet-mysql (GFZ,INGV-CNT,ODC)
            Contains:
                Networks (6):
                    IA, IB, II, IQ, IS, IV
                Stations (0):
        <BLANKLINE>
                Channels (0):
        <BLANKLINE>

        >>> fclient = FederatedClient(use_parallel=True)

        parallel request, but only one provider

        >>> INV = fclient.get_stations(network="A?", station="OK*",
        ...                           channel="?HZ", level="station",
        ...                           endtime="2016-12-31")  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        >>> print(INV)  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Inventory created at 2...Z
            Created by: IRIS WEB SERVICE: fdsnws-station | version: 1...
                    http://service.iris.edu/fdsnws/station/1/query
            Sending institution: IRIS-DMC (IRIS-DMC)
            Contains:
                Networks (1):
                    AV
                Stations (14):
                    AV.OKAK (Cape Aslik 2, Okmok Caldera, Alaska)
                    ...
                    AV.OKWR (West Rim, Okmok Caldera, Alaska)
                Channels (0):
        <BLANKLINE>

        another parallel request, this time with several providers

        >>> INV2 = fclient.get_stations(network="I?", station="AN*",
        ...                           level="network", includeoverlaps="true")
        >>> print(INV2)  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Inventory created at ...Z
            Created by: ObsPy ...
                    https://www.obspy.org
            Sending institution: IRIS-DMC,SeisComP3 (IRIS-DMC,ODC)
            Contains:
                Networks (...):
                    IU (...)
                Stations (0):
        <BLANKLINE>
                Channels (0):
        <BLANKLINE>
        """

        fed_kwargs, svc_kwargs = distribute_args(kwargs)
        fed_kwargs["includeoverlaps"] = includeoverlaps

        if "bulk" in fed_kwargs:
            #bulk = _get_bulk_string(fed_kwargs['bulk'], fed_kwargs)
            frm = self.get_routing_bulk(fed_kwargs['bulk']) #, fed_kwargs)
        else:
            frm = self.get_routing(**fed_kwargs)

        # query queries all providers
        inv, passed, failed = self.query(frm, "STATIONSERVICE", **svc_kwargs)

        if reroute and failed:
            logger.info(str(len(failed)) + " items were not retrieved, trying again," +
                        " but from any provider (while still honoring include/exclude)")
            fed_kwargs["includeoverlaps"] = True
            frm = self.get_routing_bulk(bulk=failed.text(), **fed_kwargs)
            more_inv, passed, failed = self.query(frm, "STATIONSERVICE", keep_unique=True, **svc_kwargs)
            logger.info("Retrieved {0} additional items".format(len(passed)))
            logger.info("Unable to retrieve {0} items:".format(len(failed)))
            logger.info('\n'+ str(failed))
            inv += more_inv

        # reprocess failed ?
        return inv


class FederatedRoutingManager(RoutingManager):
    """
    This class wraps the response given by the federated catalog.  Its primary
    purpose is to divide the response into parcels, each being a
    FederatedRoute containing the information required for a single request.

    Input would be the response from the federated catalog, or a similar text
    file. Output is a list of FederatedRoute objects

    >>> from obspy.clients.fdsn import Client
    >>> url = 'https://service.iris.edu/irisws/fedcatalog/1/'
    >>> params = {"net":"A*", "sta":"OK*", "cha":"*HZ"}
    >>> r = requests.get(url + "query", params=params, verify=False)
    >>> frm = FederatedRoutingManager(r.text)
    >>> print(frm)
    FederatedRoutingManager with 1 items:
    FederatedRoute for IRIS containing 0 query parameters and 26 request items
    """

    def __init__(self, textblock):
        RoutingManager.__init__(self, textblock, provider_details=PROVIDERS)  # removed kwargs

    def parse_routing(self, block_text):
        """
        create a list of FederatedRoute objects, one for each provider in response

        >>> fed_text = '''minlat=34.0
        ... level=network
        ...
        ... DATACENTER=GEOFON,http://geofon.gfz-potsdam.de
        ... DATASELECTSERVICE=http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/
        ... CK ASHT -- HHZ 2015-01-01T00:00:00 2016-01-02T00:00:00
        ...
        ... DATACENTER=INGV,http://www.ingv.it
        ... STATIONSERVICE=http://webservices.rm.ingv.it/fdsnws/station/1/
        ... HL ARG -- BHZ 2015-01-01T00:00:00 2016-01-02T00:00:00
        ... HL ARG -- VHZ 2015-01-01T00:00:00 2016-01-02T00:00:00'''
        >>> fr = FederatedRoutingManager(fed_text)
        >>> for f in fr:
        ...    print(f.provider_id + "\\n" + f.text('STATIONSERVICE'))
        GFZ
        level=network
        CK ASHT -- HHZ 2015-01-01T00:00:00.000 2016-01-02T00:00:00.000
        INGV
        level=network
        HL ARG -- BHZ 2015-01-01T00:00:00.000 2016-01-02T00:00:00.000
        HL ARG -- VHZ 2015-01-01T00:00:00.000 2016-01-02T00:00:00.000

        Here's an example parsing from the actual service:
        >>> import requests
        >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
        >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        >>> url = 'https://service.iris.edu/irisws/fedcatalog/1/'
        >>> r = requests.get(url + "query", params={"net":"IU", "sta":"ANTO", "cha":"BHZ",
        ...                  "endafter":"2013-01-01","includeoverlaps":"true",
        ...                  "level":"station"}, verify=False)
        >>> frp = FederatedRoutingManager(r.text)
        >>> for n in frp:
        ...     print(n.services["STATIONSERVICE"])
        ...     print(n.text("STATIONSERVICE"))
        http://service.iris.edu/fdsnws/station/1/
        level=station
        IU ANTO 00 BHZ 2010-11-10T21:42:00.000 2016-06-22T00:00:00.000
        IU ANTO 00 BHZ 2016-06-22T00:00:00.000 2599-12-31T23:59:59.000
        IU ANTO 10 BHZ 2010-11-11T09:23:59.000 2599-12-31T23:59:59.000
        http://www.orfeus-eu.org/fdsnws/station/1/
        level=station
        IU ANTO 00 BHZ 2010-11-10T21:42:00.000 2599-12-31T23:59:59.000
        IU ANTO 10 BHZ 2010-11-11T09:23:59.000 2599-12-31T23:59:59.000

        """

        fed_resp = []
        provider = FederatedRoute("EMPTY_EMPTY_EMPTY")
        parameters = None
        state = PreParse

        for raw_line in block_text.splitlines():
            line = FedcatResponseLine(raw_line)  # use a smarter, trimmed line
            state = state.next(line)
            if state == DatacenterItem:
                if provider.provider_id == "EMPTY_EMPTY_EMPTY":
                    parameters = provider.parameters
                provider = state.parse(line, provider)
                provider.parameters = parameters
                fed_resp.append(provider)
            else:
                state.parse(line, provider)
        if len(fed_resp) > 0 and (not fed_resp[-1].request_items):
            del fed_resp[-1]
        # TODO see if remap belongs in the FederatedClient instead
        remap = {
            "IRISDMC": "IRIS",
            "GEOFON": "GFZ",
            "SED": "ETH",
            "USPSC": "USP"
        }

        # remap provider codes because IRIS codes differ from OBSPY codes
        for dc in fed_resp:
            if dc.provider_id in remap:
                dc.provider_id = remap[dc.provider_id]
        return fed_resp

def data_to_request(data):
    """
    :rtype: FDSNBulkRequests
    :returns: representation of the data
    """
    if isinstance(data, Inventory):
        return inventory_to_bulkrequests(data)
    elif isinstance(data, Stream):
        return stream_to_bulkrequests(data)


if __name__ == '__main__':
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    import doctest
    doctest.testmod(exclude_empty=True, verbose=False)
