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
from requests.exceptions import (HTTPError, Timeout)
#from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException
from obspy.clients.fdsn.routers.routing_client import (RoutingClient,
                                                       ResponseManager)
from obspy.clients.fdsn.routers import (FederatedResponse,)
from obspy.clients.fdsn.routers.fedcatalog_parser import (PreParse,
                                                          RequestLine,
                                                          DatacenterItem)



def query_fedcatalog(targetservice, params=None, bulk=None, argdict=None):
    """
    send request to fedcatalog service, return ResponseManager object
    """
    # send request to the FedCatalog
    url = 'https://service.iris.edu/irisws/fedcatalog/1/'
    if params:
        params["targetservice"] = targetservice
        resp = requests.get(url + "query", params=params, verify=False)
    elif bulk:
        bulky = "\n".join(["targetservice=" + targetservice + "\n", bulk])
        resp = requests.post(url + "query", data=bulky, verify=False)
    else:
        raise RuntimeError("Either params or bulk must be specified")
    resp.raise_for_status()
    return resp


def inv2set(inv, level):
    """
    used to quickly decide what exists and what doesn't
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

class FedcatalogProviderMetadata(object):
    """
    Class containing datacenter details retrieved from the fedcatalog service

    keys: name, website, lastupdate, serviceURLs {servicename:url,...},
    location, description

    hopefully threadsafe
    """

    def __init__(self):
        self.provider_metadata = None
        self.provider_index = None
        self.lock = Lock()
        self.failed_refreshes = 0

    def get_names(self):
        """
        return list of provider names
        """
        with self.lock:
            if not self.provider_metadata:
                self.refresh()
        if not self.provider_index:
            return None
        else:
            return self.provider_index.keys()

    def get(self, name, detail=None):
        """
        get a datacenter property

        :type name: str
        :param name: provider name. such as IRISDMC, ORFEUS, etc.
        :type detail: str
        :param detail: property of interest.  eg, one of ('name', 'website',
        'lastupdate', 'serviceURLs', 'location', 'description').
        if no detail is provided, then the entire dict for the requested provider
        will be returned
        """
        with self.lock:
            if not self.provider_metadata:
                self.refresh()
        if not self.provider_metadata:
            return None
        else:
            if detail:
                return self.provider_metadata[self.provider_index[name]][
                    detail]
            else:
                return self.provider_metadata[self.provider_index[name]]

    def refresh(self, force=False):
        """
        retrieve provider profile from fedcatalog service

        >>> fpm = FedcatalogProviderMetadata()
        >>> # fpm.refresh(force=True)
        >>> fpm.get_names() #doctest: +ELLIPSIS
        dict_keys(['...'])

        :type force: bool
        :param force: attempt to retrieve data even if too many failed attempts
        """
        if force or self.failed_refreshes < 4:
            try:
                url = 'https://service.iris.edu/irisws/fedcatalog/1/datacenters'
                req = requests.get(url, verify=False)
                self.provider_metadata = req.json()
                self.provider_index = {
                    v['name']: k
                    for k, v in enumerate(self.provider_metadata)
                }
                self.failed_refreshes = 0
            except:
                print(
                    "Unable to update provider profiles from fedcatalog service",
                    file=sys.stderr)
                self.failed_refreshes += 1
        else:
            print(
                "Unable to retrieve provider profiles from fedcatalog service after {0} attempts",
                self.failed_refreshes,
                file=sys.stderr)
            # problem updating

        # yuck. manually account for differences between IRIS output and OBSPY expected
        self.provider_index["IRIS"] = self.provider_index["IRISDMC"]
        self.provider_index["GFZ"] = self.provider_index["GEOFON"]
        self.provider_index["ETH"] = self.provider_index["SED"]
        self.provider_index["USP"] = self.provider_index["USPSC"]

    def pretty(self, name):
        """
        return nice text representation of service without too much details
        >>> fpm = FedcatalogProviderMetadata()
        >>> print(fpm.pretty("ORFEUS"))  #doctest: +ELLIPSIS
        ORFEUS
        The ORFEUS Data Center
        de Bilt, the Netherlands
        http://www.orfeus-eu.org
        ...M
        <BLANKLINE>
        >>> print(fpm.pretty("IRIS") == fpm.pretty("IRISDMC"))
        True
        """
        self.refresh()
        return '\n'.join(
            self.get(name, k)
            for k in
            ["name", "description", "location", "website", "lastUpdate"]) + '\n'


PROVIDER_METADATA = FedcatalogProviderMetadata()

class FederatedClient(RoutingClient):
    """
    FDSN Web service request client.

    For details see the :meth:`~obspy.clients.fdsn.client.Client.__init__()`
    method.
    >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
    >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    >>> client = FederatedClient()
    >>> inv = client.get_stations(network="I?", station="AN*", channel="*HZ") #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    IRISDMC
    The IRIS Data Management Center
    Seattle, WA, USA
    http://ds.iris.edu
    ...M
    <BLANKLINE>
    >>> print(inv)  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
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
    """

    # no __init__ function.  Values are passed through to super.

    def assign_kwargs(self, argdict):
        #TODO figure out where each of the arguments belongs
        #  to the fedrequest?  to the final service?
        fed_argdict = argdict.copy()
        service_argdict = argdict.copy()
        return fed_argdict, service_argdict

    def get_request_fn(self, target_service):
        """
        get function used to query the service
        :param target_service: string containing either 'DATASELECTSERVICE' or 'STATIONSERVICE'
        """
        fun = {"DATASELECTSERVICE": self.submit_waveform_request,
               "STATIONSERVICE": self.submit_station_request}
        try:
            return fun.get(target_service)
        except ValueError:
            valid_funs = '"' + ', '.join(fun.keys)
            raise ValueError("Expected one of " + valid_funs + " but got {0}",
                             target_service)

    def submit_station_request(self, client, req, output, failed, **kwargs):
        """
        function used to query service
        :param self: FederatedResponse
        :param output: place where retrieved data go
        :param failed: place where list of unretrieved bulk request lines go
        """
        print(PROVIDER_METADATA.pretty(req.code))
        # print (kwargs)
        # print(req.text("STATIONSERVICE"))
        try:
            data = client.get_stations_bulk(bulk=req.text("STATIONSERVICE"), **kwargs)
        except FDSNException as ex:
            failed.put(req.request_lines)
            print("Failed to retrieve data from: {0}", req.code)
            print(ex)
        else:
            output.put(data)

    def submit_waveform_request(self, client, req, output, failed, **kwargs):
        """
        function used to query service
        :param output: place where retrieved data go
        :param failed: place where list of unretrieved bulk request lines go
        """
        print(PROVIDER_METADATA.pretty(req.code))
        try:
            data = client.get_waveforms_bulk( bulk=req.text("DATASELECTSERVICE"), **kwargs)

        except FDSNException:
            failed.put(req.request_lines)
            raise
        else:
            # print(data)
            output.put(data)

    def get_waveforms_bulk(self,
                           bulk,
                           quality=None,
                           minimumlength=None,
                           longestonly=None,
                           filename=None,
                           exclude_provider=None,
                           include_provider=None,
                           includeoverlaps=False,
                           **kwargs):
        """
        
        >>> client = FederatedClient()
        >>> w, _ = client.get_waveforms_bulk(bulk="IU ANMO * ?HZ 2010-02-27T06:30:00) 2010-02-27T06:33:00")
        >>> print(w)  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        
        :type exclude_provider: str or list of str
        :param exclude_provider: Get no data from these datacenters
        :type include_provider: str or list of str
        :param include_provider: Get data only from these providers
        :type includeoverlaps: boolean
        :param includeoverlaps: retrieve same information from multiple sources
        (not recommended)
        other parameters as seen in Client.get_stations_bulk
        """

        arguments = OrderedDict(
            quality=quality,
            minimumlength=minimumlength,
            longestonly=longestonly,
            includeoverlaps=includeoverlaps,
            target_service="dataselect",
            format="request")


        fed_kwargs, svc_kwargs = self.assign_kwargs(kwargs)
        try:
            resp = query_fedcatalog("DATASELECTSERVICE", params=fed_kwargs)
        except ConnectionError:
            print("Problem connecting to fedcatalog service", file=sys.stderr)
        except HTTPError:
            print(
                "Error downloading data from fedcatalog service: " +
                str(resp.status_code),
                file=sys.stderr)
        except Timeout:
            print(
                "Timeout while waiting for a response from the fedcatalog service"
            )

        # parse the reply into an iterable object
        frm = FederatedResponseManager(
            resp.text,
            include_provider=include_provider,
            exclude_provider=exclude_provider)

        query_service = self.get_query_machine()
        data, _ = query_service(frm, "DATASELECTSERVICE", **svc_kwargs)

        # reprocess failed?
        return data

    def get_waveforms(self,
                      quality=None,
                      minimumlength=None,
                      longestonly=None,
                      filename=None,
                      exclude_provider=None,
                      include_provider=None,
                      includeoverlaps=False,
                      **kwargs):
        """
        :type exclude_provider: str or list of str
        :param exclude_provider: Get no data from these datacenters
        :type include_provider: str or list of str
        :param include_provider: Get data only from these providers
        :type includeoverlaps: boolean
        :param includeoverlaps: retrieve same information from multiple sources
        (not recommended)
        other parameters as seen in Client.get_stations_bulk
        
        >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
        >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        >>> client = FederatedClient()
        >>> from obspy.core import  UTCDateTime
        >>> t_st = UTCDateTime("2010-02-27T06:30:00")
        >>> t_ed = UTCDateTime("2010-02-27T06:33:00")
        >>> w, _ = client.get_waveforms(network="IU", station="ANMO", channel="BHZ",
        ...                             starttime=t_st, endtime=t_ed)  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        IRISDMC
        The IRIS Data Management Center
        Seattle, WA, USA
        http://ds.iris.edu
        Apr 20, 2017 5:05:58 AM
        <BLANKLINE>
        >>> print(w)  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        IU.ANMO.00.BHZ | 2010-02-27T06:30:00...Z - 2010-02-27T06:32:59...Z | 20.0 Hz, 3600 samples
        """

        fed_kwargs, svc_kwargs = self.assign_kwargs(kwargs)
        fed_kwargs["includeoverlaps"] = includeoverlaps
        try:
            resp = query_fedcatalog("DATASELECTSERVICE", params=fed_kwargs)
        except ConnectionError:
            print("Problem connecting to fedcatalog service", file=sys.stderr)
        except HTTPError:
            print(
                "Error downloading data from fedcatalog service: " +
                str(resp.status_code),
                file=sys.stderr)
        except Timeout:
            print(
                "Timeout while waiting for a response from the fedcatalog service"
            )
        

        # parse the reply into an iterable object
        frm = FederatedResponseManager(
            resp.text,
            include_provider=include_provider,
            exclude_provider=exclude_provider)

        query_service = self.get_query_machine()
        data, _ = query_service(frm, "DATASELECTSERVICE", **svc_kwargs)

        # reprocess failed?
        return data

    def get_stations_bulk(self, bulk,
                          exclude_provider=None,
                          include_provider=None,
                          includeoverlaps=False,
                          **kwargs):
        """
        :type exclude_provider: str or list of str
        :param exclude_provider: Get no data from these datacenters
        :type include_provider: str or list of str
        :param include_provider: Get data only from these providers
        :type includeoverlaps: boolean
        :param includeoverlaps: retrieve same information from multiple sources
        (not recommended)
        other parameters as seen in Client.get_stations_bulk

        >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
        >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        >>> client = FederatedClient()
        >>> bulktxt = "level=channel\\nA? OKS? * ?HZ * *"
        >>> INV = client.get_stations_bulk(
        ...                     bulktxt)  #doctest: +ELLIPSIS
        IRISDMC
        The IRIS Data Management Center
        Seattle, WA, USA
        http://ds.iris.edu...
        <BLANKLINE>
        >>> print(INV)  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
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

        fed_kwargs, svc_kwargs = self.assign_kwargs(kwargs)
        fed_kwargs["includeoverlaps"] = includeoverlaps
        # send request to the FedCatalog
        resp = query_fedcatalog("station", bulk=bulk)
        # parse the reply into an iterable object
        frm = FederatedResponseManager(resp.text,
                                       include_provider=include_provider,
                                       exclude_provider=exclude_provider)

        query_service = self.get_query_machine()
        inv, _ = query_service(frm, "STATIONSERVICE", **kwargs)

        # reprocess failed?

        return inv

    def get_stations(self,
                     exclude_provider=None,
                     include_provider=None,
                     includeoverlaps=False,
                     **kwargs):
        """
        This will be the original request for the federated station service
        :type exclude_provider: str or list of str
        :param exclude_provider: Get no data from these datacenters
        :type include_provider: str or list of str
        :param include_provider: Get data only from these providers.
        :type includeoverlaps: boolean
        :param includeoverlaps: retrieve same information from multiple sources
        (not recommended)
        other parameters as seen in Client.get_stations


        >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
        >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        >>> client = FederatedClient()
        >>> INV = client.get_stations(network="A?", station="OK*",
        ...                           channel="?HZ", level="station",
        ...                           endtime="2016-12-31")  #doctest: +ELLIPSIS
        IRISDMC
        The IRIS Data Management Center
        Seattle, WA, USA
        http://ds.iris.edu...
        <BLANKLINE>
        >>> print(INV)  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Inventory created at 2...Z
            Created by: IRIS WEB SERVICE: fdsnws-station | version: 1...
                    http://service.iris.edu/fdsnws/station/1/query
            Sending institution: IRIS-DMC (IRIS-DMC)
            Contains:
                Networks (1):
                    AV
                Stations (14):
                    AV.OKAK (Cape Aslik 2, Okmok Caldera, Alaska)
                    AV.OKCD (Cone D, Okmok Caldera, Alaska)
                    AV.OKCE (Cone E, Okmok Caldera, Alaska)
                    AV.OKCF (Cone F, Okmok Caldera, Alaska)
                    AV.OKER (East Rim, Okmok Caldera, Alaska)
                    AV.OKFG (Fort Glenn, Okmok Caldera, Alaska)
                    AV.OKID (Mount Idak, Okmok Caldera, Alaska)
                    AV.OKNC (New Cone D, Okmok Caldera, Alaska)
                    AV.OKRE (Reindeer Point, Okmok Caldera, Alaska)
                    AV.OKSO (South, Okmok Caldera, Alaska)
                    AV.OKSP (Steeple Point, Okmok Caldera, Alaska)
                    AV.OKTU (Mount Tulik, Okmok Caldera, Alaska)
                    AV.OKWE (Weeping Wall, Okmok Caldera, Alaska)
                    AV.OKWR (West Rim, Okmok Caldera, Alaska)
                Channels (0):
        <BLANKLINE>
        
        >>> INV2 = client.get_stations(network="I?", station="A*",
        ...                           level="network", exclude_provider=["IRISDMC","IRIS","IRIS-DMC"])  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        GEOFON
        The GEOFON Program
        Potsdam, Germany
        http://geofon.gfz-potsdam.de
        ...M
        <BLANKLINE>
        INGV
        The Italian National Institute of Geophysics and Volcanology
        Rome, Italy
        http://www.ingv.it
        ...M
        <BLANKLINE>
        ORFEUS
        The ORFEUS Data Center
        de Bilt, the Netherlands
        http://www.orfeus-eu.org
        ...AM
        <BLANKLINE>
        >>> print(INV2)  #doctest: +ELLIPSIS
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
        """

        fed_kwargs, svc_kwargs = self.assign_kwargs(kwargs)
        fed_kwargs["includeoverlaps"] = includeoverlaps

        try:
            resp = query_fedcatalog("station", params=fed_kwargs)
        except ConnectionError:
            print("Problem connecting to fedcatalog service", file=sys.stderr)
            raise
        except HTTPError:
            print(
                "Error downloading data from fedcatalog service: " +
                str(resp.status_code),
                file=sys.stderr)
            raise
        except Timeout:
            print(
                "Timeout while waiting for a response from the fedcatalog service"
            )
            raise

        frm = FederatedResponseManager(resp.text,
                                       include_provider=include_provider,
                                       exclude_provider=exclude_provider)

        query_service = self.get_query_machine() 
        # TODO check to make sure svc_kwargs actually are passed along from fedcatalog
        inv, _ = query_service(frm, "STATIONSERVICE", **svc_kwargs)

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
    >>> frm = FederatedResponseManager(r.text, include_provider=["IRIS", "IRISDMC"])
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
    doctest.testmod(exclude_empty=True)
