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
from obspy.clients.fdsn.routers.routing_client import (RoutingClient, ResponseManager)
from obspy.clients.fdsn.routers.fedcatalog_response_parser import(FederatedResponse, PreParse,
                                                                  RequestLine, DatacenterItem)

class FedcatalogProviderMetadata(object):
    '''Class containing datacenter details retrieved from the fedcatalog service
    keys: name, website, lastupdate, serviceURLs {servicename:url,...}, location, description

    hopefully threadsafe
    '''
    def __init__(self):
        self.provider_metadata = None
        self.provider_index = None
        self.lock = Lock()
        self.failed_refreshes = 0

    def get_names(self):
        '''return list of provider names'''
        with self.lock:
            if not self.provider_metadata:
                self.refresh()
        if not self.provider_index:
            return None
        else:
            return self.provider_index.keys()

    def get(self, name, detail=None):
        '''get a datacenter property
        :type name: str
        :param name: provider name. such as IRISDMC, ORFEUS, etc.
        :type detail: str
        :param detail: property of interest.  eg, one of ('name', 'website', 'lastupdate',
        serviceURLs', 'location', 'description').  if no detail is provided, then the entire dict
        for the requeted datacenter will be provided
        '''
        with self.lock:
            if not self.provider_metadata:
                self.refresh()
        if not self.provider_metadata:
            return None
        else:
            if detail:
                return self.provider_metadata[self.provider_index[name]][detail]
            else:
                return self.provider_metadata[self.provider_index[name]]

    def refresh(self, force=False):
        '''retrieve provider profile from fedcatalog service
        :type force: bool
        :param force: attempt to retrieve data even if too many failed attempts
        '''
        if force or self.failed_refreshes < 4:
            try:
                req = requests.get('https://service.iris.edu/irisws/fedcatalog/1/datacenters',
                                   verify=False)
                self.provider_metadata = req.json()
                self.provider_index = {v['name']:k for k, v in enumerate(self.provider_metadata)}
                self.failed_refreshes = 0
            except:
                print("Unable to update provider profiles from fedcatalog service", file=sys.stderr)
                self.failed_refreshes += 1
        else:
            print("Unable to retrieve provider profiles from fedcatalog service after {0} attempts",
                  self.failed_refreshes, file=sys.stderr)
            # problem updating

    def pretty(self, name):
        ''' return nice text representation of service without too much details'''
        return '\n'.join(self.get(name, k) for k in ["name", "description", "location", "website",
                                                     "lastUpdate"])

PROVIDER_METADATA = FedcatalogProviderMetadata()

def query_fedcatalog(targetservice, params=None, bulk=None, argdict=None):
    ''' send request to fedcatalog service, return ResponseManager object'''
    # send request to the FedCatalog
    url = 'https://service.iris.edu/irisws/fedcatalog/1/'
    params["targetservice"] = targetservice
    if params:
        resp = requests.get(url + "query", params=params, verify=False)
    elif bulk:
        resp = requests.get(url + "query", data=bulk, verify=False)
    else:
        raise RuntimeError("Either params or bulk must be specified")
    resp.raise_for_status()
    return resp

def inv2set(inv, level):
    '''inv2x_set(inv) will be used to quickly decide what exists and what doesn't'''

    converter = {
        "channel": channel_set,
        "response": channel_set,
        "station": station_set,
        "network": network_set
    }

    def channel_set(inv):
        'return a set containing string representations of an inv object'
        return {
            n.code + "." + s.code + "." + c.location_code + "." + c.code
            for n in inv for s in n for c in s
        }

    def station_set(inv):
        'return a set containing string representations of an inv object'
        return {n.code + "." + s.code for n in inv for s in n}

    def network_set(inv):
        'return a set containing string representations of an inv object'
        return {n.code for n in inv}

    return converter[level](inv)

#converters used to make comparisons between inventory items and requests


class FederatedClient(RoutingClient):
    """
    FDSN Web service request client.

    For details see the :meth:`~obspy.clients.fdsn.client.Client.__init__()`
    method.
    >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
    >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    >>> client = FederatedClient()
    >>> inv = client.get_stations(network="I?", station="AN*", channel="*HZ")
    >>> print(inv)
    """

    def get_waveforms_bulk(self, bulk, quality=None,
                           minimumlength=None,
                           longestonly=None,
                           filename=None,
                           exclude_provider=None,
                           include_provider=None,
                           includeoverlaps=False,
                           **kwargs):
        '''
        :param exclude_provider: Avoids getting data from datacenters with specified ID code(s)
        :param include_provider: limit retrieved data to  datacenters with specified ID code(s)
        :param includeoverlaps: For now, simply leave this False. It will confuse the program.
        other parameters as seen in Client.get_stations_bulk
        '''

        arguments = OrderedDict(
            quality=quality,
            minimumlength=minimumlength,
            longestonly=longestonly,
            includeoverlaps=includeoverlaps,
            target_service="dataselect",
            format="request"
        )

        #bulk = self._get_bulk_string(bulk, arguments)

        # send request to the FedCatalog
        try:
            resp = query_fedcatalog("dataselect", bulk=bulk)
        except ConnectionError:
            pass
        except HTTPError:
            pass
        except Timeout:
            pass

        # parse the reply into an iterable object
        frm = FederatedResponseManager(
            resp.text,
            include_provider=include_provider,
            exclude_provider=exclude_provider)

        inv, _ = frm.parallel_service_query("DATASELECTSERVICE", **kwargs)
        return inv

    def get_waveforms(self, quality=None,
                      minimumlength=None,
                      longestonly=None,
                      filename=None,
                      exclude_provider=None,
                      include_provider=None,
                      includeoverlaps=False,
                      **kwargs):
        '''
        :param exclude_provider: Avoids getting data from datacenters with specified ID code(s)
        :param include_provider: limit retrieved data to  datacenters with specified ID code(s)
        :param includeoverlaps: For now, simply leave this False. It will confuse the program.
        other parameters as seen in Client.get_stations_bulk
        '''

        # send request to the FedCatalog
        resp = query_fedcatalog("dataselect", params=kwargs)
        # parse the reply into an iterable object
        frm = FederatedResponseManager(
            resp.text,
            include_provider=include_provider,
            exclude_provider=exclude_provider)

        inv, _ = frm.parallel_service_query("DATASELECTSERVICE", **kwargs)
        return inv

    def get_stations_bulk(self,
                          exclude_provider=None,
                          include_provider=None,
                          includeoverlaps=False,
                          bulk=None,
                          **kwargs):
        '''
        :param exclude_provider: Avoids getting data from datacenters with specified ID code(s)
        :param include_provider: limit retrieved data to  datacenters with specified ID code(s)
        :param includeoverlaps: For now, simply leave this False. It will confuse the program.
        other parameters as seen in Client.get_stations_bulk
        '''

        assert bulk, "No bulk request provided"
        # send request to the FedCatalog
        resp = query_fedcatalog("station", bulk=bulk)
        # parse the reply into an iterable object
        frm = FederatedResponseManager(
            resp.text,
            include_provider=include_provider,
            exclude_provider=exclude_provider)

        inv, _ = frm.parallel_service_query("STATIONSERVICE", **kwargs)
        return inv

    def get_stations(self,
                     exclude_provider=None,
                     include_provider=None,
                     includeoverlaps=False,
                     **kwargs):
        '''
        This will be the original request for the federated station service
        :param exclude_provider: Avoids getting data from datacenters with specified ID code(s)
        :param include_provider: limit retrieved data to  datacenters with specified ID code(s)
        :param includeoverlaps: For now, simply leave this False. It will confuse the program.
        other parameters as seen in Client.get_stations

        Warning: If you specify "include_provider", you may miss data that the provider holds,
                but isn't the primary source
        Warning: If you specify "exclude_provider", you may miss data for which that provider
                is the primary source

        >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
        >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        >>> client = FederatedClient()
        >>> INV = client.get_stations(network="A?", station="OK*", channel="?HZ", level="station",
        ...                    endtime="2016-12-31")
        '''

        try:
            resp = query_fedcatalog("station", params=kwargs)
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

        frm = FederatedResponseManager(
            resp.text,
            include_provider=include_provider,
            exclude_provider=exclude_provider)

        # prepare the file if one is specified
        inv, _ = frm.parallel_service_query("STATIONSERVICE", **kwargs)

        # level = kwargs["level"]
        # successful, failed = request_exists_in_inventory(inv, datac.request_lines, level)
        # all_inv_set = inv2set(inv, level) if not all_inv else all_inv_set.union(inv2set(inv, level))
        # resubmit unsuccessful requests to fedservice with includeoverlaps
        return inv


class FederatedResponseManager(ResponseManager):
    """
    This class wraps the response given by the federated catalog.  Its primary purpose is to
    divide the response into parcels, each being a FederatedResponse containing the information
    required for a single request.
    Input would be the response from the federated catalog, or a similar text file. Output is a list
    of FederatedResponse objects

    >>> from obspy.clients.fdsn import Client
    >>> url = 'https://service.iris.edu/irisws/fedcatalog/1/'
    >>> r = requests.get(url + "query", params={"net":"A*", "sta":"OK*", "cha":"*HZ"}, verify=False)
    >>> frm = FederatedResponseManager(r.text, include_provider=["IRIS", "IRISDMC"])
    >>> # frm = frm.subset_requests("IRIS")
    >>> print(frm)
    >>> data, retry = frm.parallel_service_query('STATIONSERVICE')
    >>> print(data)
    >>> print(retry)
    """

    def __init__(self, textblock, **kwargs):
        
        super(FederatedResponseManager, self).__init__(textblock, **kwargs)

    def parse_response(self, block_text):
        '''create a list of FederatedResponse objects, one for each provider in response
            >>> fed_text = """minlat=34.0
            ... level=network
            ...
            ... DATACENTER=GEOFON,http://geofon.gfz-potsdam.de
            ... DATASELECTSERVICE=http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/
            ... CK ASHT -- HHZ 2015-01-01T00:00:00 2016-01-02T00:00:00
            ...
            ... DATACENTER=INGV,http://www.ingv.it
            ... STATIONSERVICE=http://webservices.rm.ingv.it/fdsnws/station/1/
            ... HL ARG -- BHZ 2015-01-01T00:00:00 2016-01-02T00:00:00
            ... HL ARG -- VHZ 2015-01-01T00:00:00 2016-01-02T00:00:00"""
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

        '''
        fed_resp = []
        provider = FederatedResponse("EMPTY_EMPTY_EMPTY")
        parameters = None
        state = PreParse

        for raw_line in block_text.splitlines():
            line = RequestLine(raw_line)  #use a smarter, trimmed line
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
    import doctest
    doctest.testmod(exclude_empty=True)
