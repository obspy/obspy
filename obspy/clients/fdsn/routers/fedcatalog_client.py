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
import sys
from collections import OrderedDict
from threading import Lock
import requests
# from requests.exceptions import (HTTPError, Timeout)
from obspy.clients.fdsn.header import FDSNException
from obspy.clients.fdsn.routers.routing_client import (RoutingClient,
                                                       ResponseManager)
from obspy.clients.fdsn.routers import (FederatedResponse,)
from obspy.clients.fdsn.routers.fedcatalog_parser import (PreParse,
                                                          RequestLine,
                                                          DatacenterItem)


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

def assign_kwargs(argdict):
    """
    divide a dictionary's keys between fedcatalog and fdsnservice

    :type argdict: dict
    :param argdict: 
    :rtype:
    :return: tuple of dictionaries fedcat_kwargs, fdsn_kwargs
    """
    #TODO figure out where each of the arguments belongs
    #  to the fedrequest?  to the final service?

    fedcatalog_params= ('')
    service_params= ('')
    # fedrequest gets almost all arguments
    fed_argdict = argdict.copy()
    service_argdict = argdict.copy()
    return fed_argdict, service_argdict

class FedcatalogProviders(object):
    """
    Class containing datacenter details retrieved from the fedcatalog service

    keys: name, website, lastupdate, serviceURLs {servicename:url,...},
    location, description

    >>> prov = FedcatalogProviders()
    >>> print(prov.pretty('IRISDMC'))  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    IRISDMC
    The IRIS Data Management Center
    Seattle, WA, USA
    http://ds.iris.edu
    ...M
    <BLANKLINE>

    """

    def __init__(self):
        self._providers = dict()
        self._lock = Lock()
        self._failed_refreshes = 0

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
        with self._lock:
            if self._failed_refreshes > 3 and not force:
                print(
                    "Unable to retrieve provider profiles from"
                    " fedcatalog service after {0} attempts",
                    self._failed_refreshes,
                    file=sys.stderr)

            try:
                url = 'https://service.iris.edu/irisws/fedcatalog/1/datacenters'
                req = requests.get(url, verify=False)
                self._providers = {v['name']: v for v in req.json()}
                self._failed_refreshes = 0
            except:
                print(
                    "Unable to update provider profiles from fedcatalog service",
                    file=sys.stderr)
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
        ORFEUS
        The ORFEUS Data Center
        de Bilt, the Netherlands
        http://www.orfeus-eu.org
        ...M
        <BLANKLINE>
        >>> print(providers.pretty("IRIS") == providers.pretty("IRISDMC"))
        True
        """
        if not self._providers:
            self.refresh()
        if not name in self._providers:
            return ""
        fields = ("name", "description", "location", "website", "lastUpdate")
        return '\n'.join(self._providers[name][k] for k in fields) + '\n'


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
    IRISDMC
    The IRIS Data Management Center
    Seattle, WA, USA
    http://ds.iris.edu
    ...
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

    .. Warning: if output is sent directly to a file, then the success
                status will not be checked beyond gross failures (no data, no response, timeout)
    """

    def __str__(self):
        ret = "Federated Catalog Routing Client"
        return ret

    def __repr__(self):
        return "Federated Catalog Routing Client v{0}".format(self.version)

    def query_fedcatalog(self, params=None, bulk=None):
        """
        send query to the fedcatalog service

        >>> client = FederatedClient()
        >>> params={"station":"ANTO","includeoverlaps":"true"}
        >>> frm = client.query_fedcatalog(params=params)
        >>> for f in frm:
        ...   print(f)  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        IRIS, with 1... lines
        ORFEUS, with ... lines

        :type params:
        :param params:
        :type bulk:
        :param bulk:
        :rtype: :class:`~obspy.clients.fdsn.routers.FederatedResponseManager`
        :return: parsed response from the FedCatalog service
        """
        if params is None and bulk is None:
            raise ValueError("Both params and bulk are empty")
        elif bool(params) and bool(bulk):
            raise ValueError("Use params OR bulk, but not both")
        if params:
            resp = requests.get(FEDCATALOG_URL + "query", params=params, verify=False)
        else:
            resp = requests.post(FEDCATALOG_URL + "query", data=bulk, verify=False)

        resp.raise_for_status()


        frm = FederatedResponseManager(resp.text)
        return frm

    def request_something(self, client, service, req, output, failed, filename=None, **kwargs):
        """
        function used to query FDSN webservice using 
        :meth:`~obspy.clients.fdsn.client.Client.get_waveforms_bulk` or
        :meth:`~obspy.clients.fdsn.client.Client.get_stations_bulk`

        :type client:
        :param client:
        :type service:
        :param service:
        :type req: 
        :param req:
        :type output: container accepting "put"
        :param output: place where retrieved data go
        :type failed: contdainer accepting "put"
        :param failed: place where list of unretrieved bulk request lines go
        :param filename:
        :type **kwargs:
        :param **kwargs:
        """
        
        print(PROVIDERS.pretty(req.code))

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
                with open(filename, 'ab+') as f:
                    raise ValueError('not expected to be here')
                    get_bulk(bulk=req.text(service), filename=f, **kwargs)
            elif filename:
                # likely a pointer to somewhere. just let it go through without collecting the data
                raise ValueError('not expected to be here either')
                get_bulk(bulk=req.text(service), filename=filename, **kwargs)
            else:
                data = get_bulk(bulk=req.text(service), filename=filename, **kwargs)
        except FDSNException as ex:
            failed.put(req.request_lines)
            print("Failed to retrieve data from: {0}", req.code)
            print(ex)
            raise
        else:
            output.put(data)

    def get_waveforms_bulk(self,
                           bulk,
                           quality=None,
                           minimumlength=None,
                           longestonly=None,
                           filename=None,
                           includeoverlaps=False,
                           **kwargs):
        """
        retrieve waveforms from data providers via POST request to the Fedcatalog service

        >>> client = FederatedClient()
        >>> bulkreq = "IU ANMO * ?HZ 2010-02-27T06:30:00 2010-02-27T06:33:00"
        >>> tr = client.get_waveforms_bulk(bulk=bulkreq)
        ...        #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        IRISDMC
        The IRIS Data Management Center
        ...
        >>> print(tr)  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        6 Trace(s) in Stream:
        IU.ANMO.00.BHZ | 2010-02-27T06:30... | 20.0 Hz, 3600 samples
        IU.ANMO.00.LHZ | 2010-02-27T06:30... | 1.0 Hz, 180 samples
        ...
        IU.ANMO.10.VHZ | 2010-02-27T06:30... | 0.1 Hz, 18 samples

        :type includeoverlaps: boolean
        :param includeoverlaps: retrieve same information from multiple sources
        (not recommended)
        other parameters as seen in :meth:`~obspy.fdsn.clients.Client.get_waveforms_bulk`
        and :meth:`~obspy.fdsn.clients.Client.get_stations_bulk`
        :rtype: :class:`~obspy.core.stream.Stream`
        :return:
        """

        arguments = OrderedDict(
            quality=quality,
            minimumlength=minimumlength,
            longestonly=longestonly,
            includeoverlaps=includeoverlaps,
            target_service="dataselect",
            format="request")


        fed_kwargs, svc_kwargs = assign_kwargs(kwargs)
        frm = self.query_fedcatalog(bulk=bulk)

        data, _ = self.query(frm, "DATASELECTSERVICE", **svc_kwargs)

        # reprocess failed?
        return data

    def get_waveforms(self,
                      quality=None,
                      minimumlength=None,
                      longestonly=None,
                      filename=None,
                      includeoverlaps=False,
                      **kwargs):
        """
        retrieve waveforms from data providers via GET request to the Fedcatalog service

        :type includeoverlaps: boolean
        :param includeoverlaps: retrieve same information from multiple sources
        (not recommended)
        other parameters as seen in :meth:`~obspy.fdsn.clients.Client.get_waveforms`
        :rtype: :class:`~obspy.core.stream.Stream`
        :return:

        >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
        >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        >>> client = FederatedClient()
        >>> from obspy.core import  UTCDateTime
        >>> t_st = UTCDateTime("2010-02-27T06:30:00")
        >>> t_ed = UTCDateTime("2010-02-27T06:33:00")
        >>> tr = client.get_waveforms(network="IU", station="ANMO", channel="BHZ",
        ...                             starttime=t_st, endtime=t_ed)  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        IRISDMC
        The IRIS Data Management Center
        ...
        >>> print(tr)  #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        2 Trace(s) in Stream:
        IU.ANMO.00.BHZ | 2010-02-27T06:30:00... 20.0 Hz, 3600 samples
        IU.ANMO.10.BHZ | 2010-02-27T06:30:00... 40.0 Hz, 7200 samples
        """

        fed_kwargs, svc_kwargs = assign_kwargs(kwargs)
        fed_kwargs["includeoverlaps"] = includeoverlaps
        frm = self.query_fedcatalog(params=fed_kwargs)

        data, _ = self.query(frm, "DATASELECTSERVICE", **svc_kwargs)

        # reprocess failed?
        return data

    def get_stations_bulk(self, bulk,
                          includeoverlaps=False,
                          **kwargs):
        """
        retrieve station metadata from data providers via POST request to the Fedcatalog service

        :type includeoverlaps: boolean
        :param includeoverlaps: retrieve same information from multiple sources
        (not recommended)
        :rtype: :class:`~obspy.core.inventory.inventory.Inventory`
        :return:
        other parameters as seen in :meth:`~obspy.fdsn.clients.Client.get_stations_bulk`

        >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
        >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        >>> client = FederatedClient()
        >>> bulktxt = "level=channel\\nA? OKS? * ?HZ * *"
        >>> INV = client.get_stations_bulk(
        ...                     bulktxt)  #doctest: +ELLIPSIS
        IRISDMC
        The IRIS Data Management Center
        ...
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

        fed_kwargs, svc_kwargs = assign_kwargs(kwargs)
        fed_kwargs["includeoverlaps"] = includeoverlaps
        # send request to the FedCatalog
        frm = self.query_fedcatalog(bulk=bulk)

        inv, _ = self.query(frm, "STATIONSERVICE", **svc_kwargs)

        # reprocess failed?

        return inv

    def get_stations(self,
                     includeoverlaps=False,
                     **kwargs):
        """
        retrieve station metadata from data providers via GET request to the Fedcatalog service

        :type includeoverlaps: boolean
        :param includeoverlaps: retrieve same information from multiple sources
        (not recommended)
        other parameters as seen in :meth:`~obspy.fdsn.clients.Client.get_stations`
        :rtype: :class:`~obspy.core.inventory.inventory.Inventory`
        :return:

        >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
        >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        >>> fclient = FederatedClient()
        >>> INV = fclient.get_stations(network="A?", station="OK*",
        ...                           channel="?HZ", level="station",
        ...                           endtime="2016-12-31")  #doctest: +ELLIPSIS
        IRISDMC
        The IRIS Data Management Center
        ...
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
        >>> keep_out = ["IRISDMC","IRIS","IRIS-DMC"]
        >>> fclient.exclude_provider = keep_out
        >>> INV2 = fclient.get_stations(network="I?", station="A*",
        ...                           level="network")
        ...                           #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        GEOFON
        The GEOFON Program
        Potsdam, Germany
        ...
        INGV
        The Italian National Institute of Geophysics and Volcanology
        Rome, Italy
        ...
        ORFEUS
        The ORFEUS Data Center
        de Bilt, the Netherlands
        ...
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
        >>> INV = fclient.get_stations(network="A?", station="OK*",
        ...                           channel="?HZ", level="station",
        ...                           endtime="2016-12-31")  #doctest: +ELLIPSIS
        IRISDMC
        The IRIS Data Management Center
        ...
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
        """

        fed_kwargs, svc_kwargs = assign_kwargs(kwargs)
        fed_kwargs["includeoverlaps"] = includeoverlaps

        frm = self.query_fedcatalog(params=fed_kwargs)


        # TODO check to make sure svc_kwargs actually are passed along from fedcatalog
        inv, _ = self.query(frm, "STATIONSERVICE", **svc_kwargs)

        # reprocess failed ?
        return inv


class FederatedResponseManager(ResponseManager):
    """
    This class wraps the response given by the federated catalog.  Its primary
    purpose is to divide the response into parcels, each being a
    FederatedResponse containing the information required for a single request.

    Input would be the response from the federated catalog, or a similar text
    file. Output is a list of FederatedResponse objects

    >>> from obspy.clients.fdsn import Client
    >>> url = 'https://service.iris.edu/irisws/fedcatalog/1/'
    >>> params = {"net":"A*", "sta":"OK*", "cha":"*HZ"}
    >>> r = requests.get(url + "query", params=params, verify=False)
    >>> frm = FederatedResponseManager(r.text)
    >>> print(frm)
    FederatedResponseManager with 1 items:
    IRIS, with 26 lines
    """

    def __init__(self, textblock, **kwargs):
        ResponseManager.__init__(self, textblock, **kwargs)

    def parse_response(self, block_text):
        """
        create a list of FederatedResponse objects, one for each provider in response

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
        >>> fr = FederatedResponseManager(fed_text)
        >>> for f in fr:
        ...    print(f.code + "\\n" + f.text('STATIONSERVICE'))
        GFZ
        level=network
        CK ASHT -- HHZ 2015-01-01T00:00:00 2016-01-02T00:00:00
        INGV
        level=network
        HL ARG -- BHZ 2015-01-01T00:00:00 2016-01-02T00:00:00
        HL ARG -- VHZ 2015-01-01T00:00:00 2016-01-02T00:00:00

        Here's an example parsing from the actual service:
        >>> import requests
        >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
        >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        >>> url = 'https://service.iris.edu/irisws/fedcatalog/1/'
        >>> r = requests.get(url + "query", params={"net":"IU", "sta":"ANTO", "cha":"BHZ",
        ...                  "endafter":"2013-01-01","includeoverlaps":"true",
        ...                  "level":"station"}, verify=False)
        >>> frp = FederatedResponseManager(r.text)
        >>> for n in frp:
        ...     print(n.services["STATIONSERVICE"])
        ...     print(n.text("STATIONSERVICE"))
        http://service.iris.edu/fdsnws/station/1/
        level=station
        IU ANTO 00 BHZ 2010-11-10T21:42:00 2016-06-22T00:00:00
        IU ANTO 00 BHZ 2016-06-22T00:00:00 2599-12-31T23:59:59
        IU ANTO 10 BHZ 2010-11-11T09:23:59 2599-12-31T23:59:59
        http://www.orfeus-eu.org/fdsnws/station/1/
        level=station
        IU ANTO 00 BHZ 2010-11-10T21:42:00 2599-12-31T23:59:59
        IU ANTO 10 BHZ 2010-11-11T09:23:59 2599-12-31T23:59:59

        """

        fed_resp = []
        provider = FederatedResponse("EMPTY_EMPTY_EMPTY")
        parameters = None
        state = PreParse

        for raw_line in block_text.splitlines():
            line = RequestLine(raw_line)  # use a smarter, trimmed line
            state = state.next(line)
            if state == DatacenterItem:
                if provider.code == "EMPTY_EMPTY_EMPTY":
                    parameters = provider.parameters
                provider = state.parse(line, provider)
                provider.parameters = parameters
                fed_resp.append(provider)
            else:
                state.parse(line, provider)
        if len(fed_resp) > 0 and (not fed_resp[-1].request_lines):
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
            if dc.code in remap:
                dc.code = remap[dc.code]
        return fed_resp


if __name__ == '__main__':
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    import doctest
    doctest.testmod(exclude_empty=True, verbose=True)
