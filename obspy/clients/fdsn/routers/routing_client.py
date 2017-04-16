#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FDSN Web service client for ObsPy.
"""

from obspy.clients.fdsn import Client

class RoutingClient(Client):
    """
    This class serves as the user-facing layer for federated requests, and uses
    the Client's methods to communicate with each data center.  Where possible,
    it will also leverage the Client's methods to interact with the federated catalog service.
    The federated catalog's response is passed to the ResponseManager.
    The ResponseManager is then repeatedly queried for datacenter/url/bulk-request parcels,
    which are each routed to the appropriate datacenter using either Client.get_stations_bulk
    or Client.get_waveforms_bulk.  As each parcel of data is requested, the datacenter
    is displayed to console. As each request is fulfilled, then a summary of retrieved data
     is displayed.
    Upon completion, all the waveforms or inventory (station) data are returned in a
    single list /structure as though they were all requested from the same source. It
    appears that the existing obspy.core.inventory module will appropriately attribute
    each station to a datacenter as it is downloaded, but once they are merged into the
    same inventory object, individual station:datacenter identity is lost.
    """
    def get_fedcatalog_index(**kwargs):
        print("getting federated catalog index")

        url = self._create_url_from_parameters(
            "fedcatalog", DEFAULT_PARAMETERS['station'], kwargs)
        # DEFAULT_PARAMETERS probably needs a federated catalog version
        
        data_stream = self._download(url)
        data_stream.seek(0, 0)
        
        fedcatalog_index = read_fedcatalog_response(data_stream)
        data_stream.close()
        return fedcatalog_index

    def get_fedcatalog_index_bulk(**kwargs):
        r"""
        Query the station service of the client. Bulk request.

        For details see the :meth:`~obspy.clients.fdsn.client.Client.get_stations_bulk()`
        method.
        """
        if "fedcatalog" not in self.services:
            msg = "The current client does not have a dataselect service."
            raise ValueError(msg)

        arguments = OrderedDict(
            quality=quality,
            minimumlength=minimumlength,
            longestonly=longestonly,
            level=level,
            includerestriced=includerestricted,
            includeavailability=includeavailability
        )
        bulk = self._get_bulk_string(bulk, arguments)

        url = self._build_url("fedcatalog", "query")

        data_stream = self._download(url,
                                     data=bulk.encode('ascii', 'strict'))
        data_stream.seek(0, 0)
  
        fedcatalog_index = read_fedcatalog_response(data_stream)
        data_stream.close()
        return fedcatalog_index

class ResponseManager(object):
    """
    This class will wrap the response given by routers.  Its primary purpose is to
    divide the response into parcels, each being an XYZResponse containing the information
    required for a single request.
    Input would be the response from the routing service, or a similar text file Output is a list
    of FederatedResponse objects

    """
    def __init__(self, textblock, include_datacenter=None, exclude_datacenter=None):
        '''
        :param textblock: input is returned text from routing service
        '''

        self.responses = self.parse_response(textblock)
        if include_datacenter or exclude_datacenter:
            self.responses = self.subset_requests(self.responses)

    def __iter__(self):
        return self.responses.__iter__()

    def __len__(self):
        return len(self.responses)

    def __str__(self):
        responsestr = "\n  ".join([str(x) for x in self.responses])
        towrite = "ResponseManager with " + str(len(self)) + " items:\n" +responsestr
        return towrite

    def parse_response(self, parameter_list):
        except
        pass

    def funcname(self, parameter_list):
        pass

    def get_request(self, code, get_multiple=False):
        '''retrieve the response for a particular datacenter, by code
        if get_multiple is true, then a list will be returned with 0 or more
        FederatedResponse objects that meet the criteria.  otherwise, the first
        matching FederatedResponse will be returned

        Set up sample data:
        >>> fedresps = [FederatedResponse('IRIS'), FederatedResponse('SED'),
        ...             FederatedResponse('RESIF'), FederatedResponse('SED')]

        Test methods that return multiple FederatedResponse objects
        >>> get_datacenter_request(fedresps, 'SED')
        SED
        <BLANKLINE>
        >>> get_request(fedresps, 'SED', get_multiple=True)
        [SED
        , SED
        ]
        '''
        if get_multiple:
            return [resp for resp in self.responses if resp.code == code]
        for resp in self.responses:
            if resp.code == code:
                return resp
        return None

    def subset_requests(self, include_datacenter=None, exclude_datacenter=None):
        '''provide more flexibility by specifying which datacenters to include or exclude

        Set up sample data:
        >>> fedresps = [FederatedResponse('IRIS'), FederatedResponse('SED'), FederatedResponse('RESIF')]

        >>> unch = subset_requests(fedresps)
        >>> print(".".join([dc.code for dc in unch]))
        IRIS.SED.RESIF

        Test methods that return multiple FederatedResponse objects
        >>> no_sed_v1 = subset_requests(fedresps, exclude_datacenter='SED')
        >>> no_sed_v2 = subset_requests(fedresps, include_datacenter=['IRIS', 'RESIF'])
        >>> print(".".join([dc.code for dc in no_sed_v1]))
        IRIS.RESIF
        >>> ".".join([x.code for x in no_sed_v1]) == ".".join([x.code for x in no_sed_v2])
        True

        Test methods that return single FederatedResponse (still in a container, though)
        >>> only_sed_v1 = subset_requests(fedresps, exclude_datacenter=['IRIS', 'RESIF'])
        >>> only_sed_v2 = subset_requests(fedresps, include_datacenter='SED')
        >>> print(".".join([dc.code for dc in only_sed_v1]))
        SED
        >>> ".".join([x.code for x in only_sed_v1]) == ".".join([x.code for x in only_sed_v2])
        True
        '''
        if include_datacenter:
            return [resp for resp in self.responses if resp.code in include_datacenter]
        elif exclude_datacenter:
            return [resp for resp in self.responses if resp.code not in exclude_datacenter]
        else:
            return self.responses


REMAP = {"IRISDMC":"IRIS", "GEOFON":"GFZ", "SED":"ETH", "USPSC":"USP"}
class FederatedResponseManager(ResponseManager):
    """
    This class wraps the response given by the federated catalog.  Its primary purpose is to
    divide the response into parcels, each being a FederatedResponse containing the information
    required for a single request.
    Input would be the response from the federated catalog, or a similar text file. Output is a list
    of FederatedResponse objects

    >>> from obspy.clients.fdsn import Client
    >>> r = requests.get(url + "query", params={"net":"A*", "sta":"OK*", "cha":"*HZ"}, verify=False)
    >>> frm = FederatedResponseManager(r.text)
    >>> frm = frm.subset_requests("IRIS")
    >>> for req in frm:
    ...     c = req.client()
    ...     inv = c.get_stations_bulk(req.text)
    ...     print(inv)
    """
    def __init__(self, textblock):
        ResponseManager.__init__(self, textblock)

    def parse_response(self, block_text):
        '''create a list of FederatedResponse objects, one for each datacenter in response
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
            >>> fr = parse_federated_response(fed_text)
            >>> _ = [print(fr[n]) for n in range(len(fr))]
            GEOFON
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
            ...                  "endafter":"2013-01-01","includeoverlaps":"true","level":"station"},
            ...                  verify=False)
            >>> frp = parse_federated_response(r.text)
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
        datacenter = FederatedResponse("PRE_CENTER")
        parameters = None
        state = PreParse

        for raw_line in block_text.splitlines():
            line = RequestLine(raw_line) #use a smarter, trimmed line
            state = state.next(line)
            if state == DatacenterItem:
                if datacenter.code == "PRE_CENTER":
                    parameters = datacenter.parameters
                datacenter = state.parse(line, datacenter)
                datacenter.parameters = parameters
                fed_resp.append(datacenter)
            else:
                state.parse(line, datacenter)
        if len(fed_resp) > 0 and (not fed_resp[-1].request_lines):
            del fed_resp[-1]

        # remap datacenter codes because IRIS codes differ from OBSPY codes
        for dc in fed_resp:
            if dc.code in REMAP:
                dc.code = REMAP[dc.code]
        return fed_resp
