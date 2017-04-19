#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
fedcatalog_response_parser conatains the FederatedResponse class, with the supporting parsing
routines'''
from __future__ import print_function
import sys
from threading import Lock
import requests
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException
from obspy.clients.fdsn.routers.routing_response import RoutingResponse

class FedcatalogProviderMetadata(object):
    '''
    Class containing datacenter details retrieved from the fedcatalog service
    keys: name, website, lastupdate, serviceURLs {servicename:url,...}, location, description

    hopefully threadsafe
    '''
    def __init__(self):
        self.provider_metadata = None
        self.provider_index = None
        self.lock = Lock()
        self.failed_refreshes = 0

    def get_names(self):
        '''
        return list of provider names
        '''
        with self.lock:
            if not self.provider_metadata:
                self.refresh()
        if not self.provider_index:
            return None
        else:
            return self.provider_index.keys()

    def get(self, name, detail=None):
        '''
        get a datacenter property

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
        '''
        retrieve provider profile from fedcatalog service

        >>> fpm = FedcatalogProviderMetadata()
        >>> # fpm.refresh(force=True)
        >>> fpm.get_names() #doctest: +ELLIPSIS
        dict_keys(['...'])

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

        #yuck. manually account for differences between IRIS output and OBSPY expected
        self.provider_index["IRIS"] =self.provider_index["IRISDMC"]
        self.provider_index["GFZ"] =self.provider_index["GEOFON"]
        self.provider_index["ETH"] =self.provider_index["SED"]
        self.provider_index["USP"] =self.provider_index["USPSC"]

    def pretty(self, name):
        '''
        return nice text representation of service without too much details
        >>> fpm = FedcatalogProviderMetadata()
        >>> print(fpm.pretty("ORFEUS"))
        ORFEUS
        The ORFEUS Data Center
        de Bilt, the Netherlands
        http://www.orfeus-eu.org
        Apr 19, 2017 5:06:20 AM
        >>> print(fpm.pretty("IRIS") == fpm.pretty("IRISDMC"))
        True
        '''
        self.refresh()
        return '\n'.join(self.get(name, k) for k in ["name", "description", "location", "website",
                                                     "lastUpdate"])


PROVIDER_METADATA = FedcatalogProviderMetadata()
#PROVIDER_METADATA.refresh(force=True)

class FederatedResponse(RoutingResponse):
    '''
    >>> fed_resp = FederatedResponse("IRISDMC")
    >>> fed_resp.add_query_param(["lat=50","lon=20","level=cha"])
    >>> fed_resp.add_service("STATIONSERVICE","http://service.iris.edu/fdsnws/station/1/")
    >>> fed_resp.add_request_line("AI ORCD -- BHZ 2015-01-01T00:00:00 2016-01-02T00:00:00")
    >>> fed_resp.add_request_line("AI ORCD 04 BHZ 2015-01-01T00:00:00 2016-01-02T00:00:00")
    >>> print(fed_resp.text("STATIONSERVICE"))
    level=cha
    AI ORCD -- BHZ 2015-01-01T00:00:00 2016-01-02T00:00:00
    AI ORCD 04 BHZ 2015-01-01T00:00:00 2016-01-02T00:00:00
    '''

    #TODO maybe see which parameters are supported by specific service (?)
    # for example. at this exact moment in time, SoCal's dataselect won't accept quality
    pass_through_params = {
        "DATASELECTSERVICE":["longestonly", "quality", "minimumlength"],
        "STATIONSERVICE":["level", "matchtimeseries", "includeavailability",
                          "includerestricted", "format"]}

    def __init__(self, code):
        '''
        initialize a FederatedResponse
        :param code: code for the data provider
        '''
        super(FederatedResponse, self).__init__(code)
        self.parameters = []
        self.services = {}

    def add_service(self, service_name, service_url):
        '''add a service url to this response
        :param service_name: name such as STATIONSERVICE, DATASELECTSERVICE, or DATACENTER
        :param service_url: url of service, like http://service.iris.edu/fdsnws/station/1/
        '''
        self.services[service_name] = service_url

    def add_query_param(self, parameters):
        '''
        add parameters to list that may be prepended to a request
        :param parameters: strings of the form "param=value"
        >>> fedresp = FederatedResponse("ABC")
        >>> fedresp.add_query_param(["level=station","quality=D"])
        >>> fedresp.add_query_param("onceuponatime=now")
        >>> fedresp.add_query_param(RequestLine("testing=true"))
        >>> fedresp.add_query_param([RequestLine("black=white"),RequestLine("this=that")])
        >>> print(",".join(fedresp.parameters))
        level=station,quality=D,onceuponatime=now,testing=true,black=white,this=that
        '''
        if isinstance(parameters, str):
            self.parameters.append(parameters)
        elif isinstance(parameters, RequestLine):
            self.parameters.append(str(parameters))
        else:
            self.parameters.extend([str(p) for p in parameters])

    def add_request_line(self, request_line):
        '''append a single request to the list of requests
        :param request_lines: string or RequestLine that looks something like:
        NET STA LOC CHA yyyy-mm-ddTHH:MM:SS yyyy-mm-ddTHH:MM:SS
        '''
        if isinstance(request_line, (str, RequestLine)):
            self.request_lines.append(str(request_line))  # no returns expected!
        elif isinstance(request_line, (list, tuple)):
            self.request_lines.extend([str(line) for line in request_line])

    def matched_and_unmatched(self, templates, level):
        '''
        returns tuple of lines that (match, do_not_match) templates, to a level of detail specified
        by the level
        :param templates: containers containing strings NET.STA.LOC.CHA, NET.STA, or NET
        :param level: one of 'channel', 'response', 'station', 'network'

        >>> fedresp = FederatedResponse("ABC")
        >>> fedresp.add_request_line(["AB STA1 00 BHZ 2005-01-01 2005-03-24",
        ...                            "AB STA2 00 BHZ 2005-01-01 2005-03-24",
        ...                            "AB STA1 00 EHZ 2005-01-01 2005-03-24"])
        >>> print(fedresp.text("STATIONSERVICE"))
        AB STA1 00 BHZ 2005-01-01 2005-03-24
        AB STA2 00 BHZ 2005-01-01 2005-03-24
        AB STA1 00 EHZ 2005-01-01 2005-03-24
        >>> m, unm = fedresp.matched_and_unmatched(["AB.STA1"], "station")
        >>> print(m)
        ['AB STA1 00 BHZ 2005-01-01 2005-03-24', 'AB STA1 00 EHZ 2005-01-01 2005-03-24']
        >>> print(unm)
        ['AB STA2 00 BHZ 2005-01-01 2005-03-24']
        >>> m, _ = fedresp.matched_and_unmatched(["AB.STA1.00.BHZ", "AB.STA2.00.BHZ"], "channel")
        >>> print(m)
        ['AB STA1 00 BHZ 2005-01-01 2005-03-24', 'AB STA2 00 BHZ 2005-01-01 2005-03-24']
        '''
        converter_choices = {
            "channel": lambda x: ".".join(x[0:4]),
            "response": lambda x: ".".join(x[0:4]),
            "station": lambda x: ".".join(x[0:2]),
            "network": lambda x: x[0]
        }
        converter = converter_choices[level]
        matched = [line for line in self.request_lines if converter(line.split()) in templates]
        unmatched = [line for line in self.request_lines if line not in matched]
        return matched, unmatched

    def text(self, target_service):
        '''
        Return a string suitable for posting to a target service
        :param target_service: string name of target service, like 'DATASELECTSERVICE'
        '''
        reply = self.selected_common_parameters(target_service)
        reply.extend(self.request_lines)
        return "\n".join(reply)

    def selected_common_parameters(self, target_service):
        '''Return common parameters, targeted for a specific service
        This effecively filters out parameters that don't belong in a request.
        for example, STATIONSERVICE can accept level=xxx ,
        while DATASELECTSERVICE can accept longestonly=xxx
        :param target_service: string containing either 'DATASELECTSERVICE' or 'STATIONSERVICE'
        '''
        reply = []
        for good in FederatedResponse.pass_through_params[target_service]:
            reply.extend([c for c in self.parameters if c.startswith(good + "=")])
        return reply

    def __str__(self):
        if len(self) != 1:
            line_or_lines = " lines"
        else:
            line_or_lines = " line"
        return self.code + ", with " + str(len(self)) + line_or_lines

    def client(self):
        '''returns appropriate obspy.clients.fdsn.clients.Client for request'''
        try:
            client = Client(self.code)
        except Exception as ex:
            print("Problem assigning client " + self.code, file=sys.stderr)
            print(ex, file=sys.stderr)
            raise
        return client

    def get_request_fn(self, target_service):
        '''
        get function used to query the service
        :param target_service: string containing either 'DATASELECTSERVICE' or 'STATIONSERVICE'
        '''
        fun = {"DATASELECTSERVICE": self.submit_waveform_request,
               "STATIONSERVICE": self.submit_station_request}
        if not(target_service == 'waveform' or target_service == 'STATIONSERVICE'):
            raise ValueError("Expected either waveform or station, but got {0}", target_service)
        return fun.get(target_service)

    def submit_waveform_request(self, output, failed):
        '''
        function used to query service
        :param output: place where retrieved data go
        :param failed: place where list of unretrieved bulk request lines go
        '''
        try:
            client = self.client()
            print("requesting data from:" + self.code)
            #print("requesting data from:" + self.code + " : " + self.services["DATACENTER"])
            data = client.get_waveforms_bulk(bulk=self.text("DATASELECTSERVICE"))
        except FDSNException:
            failed.put(self.request_lines)
            # raise
        else:
            print(data)
            output.put(data)

    def submit_station_request(self, output, failed):
        '''
        function used to query service
        :param self: FederatedResponse
        :param output: place where retrieved data go
        :param failed: place where list of unretrieved bulk request lines go
        '''
        try:
            client = self.client()
            print(PROVIDER_METADATA.get(self.code,'description'))
            print("requesting data from:" + self.code + " : " + self.services["DATACENTER"])
            print(PROVIDER_METADATA.pretty(self.code))
            data = client.get_stations_bulk(bulk=self.text("STATIONSERVICE"))
        except FDSNException:# as ex:
            failed.put(self.request_lines)
        else:
            output.put(data)



class RequestLine(object):
    """line from federated catalog source that provides additional tests

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
    ...    x = RequestLine(y)
    ...    print("\\n".join([str([x.is_empty(), x.is_datacenter(), x.is_param(), x.is_request(),
    ...                           x.is_service()])]))
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
        'true if self is empty'
        return self.line == ""

    def is_datacenter(self):
        'true if self contains datacenter details'
        return self.line.startswith('DATACENTER=')

    def is_param(self):
        'true if self could be a param=value'
        # true for provider, services, and parameter_list
        return '=' in self.line

    def is_request(self):
        'true if self might be in proper request format for posting to web services'
        return len(self.line.split()) == 6 # and test field values?

    def is_service(self):
        'true if a parameter that might be pointing to a service'
        return self.is_param() and self.line.split("=")[0].isupper() and not self.is_datacenter()

    def __init__(self, line):
        self.line = line.strip()

    def __repr__(self):
        return self.line

    def __str__(self):
        return self.line

class ParserState(object):
    '''
    Parsers leverage the known structure of Fedcatalog's response

    PREPARSE -> [PARAMLIST | EMPTY_LINE | DATACENTER]
    PARAMLIST -> [PARAMLIST | EMPTY_LINE]
    EMPTY_LINE -> [EMPTY_LINE | DATACENTER | DONE]
    DATACENTER -> [SERVICE]
    SERVICE -> [SERVICE | REQUEST]
    REQUEST -> [REQUEST | EMPTY_LINE | DONE ]
    '''

    @staticmethod
    def parse(line, this_response):
        '''abstract'''
        raise NotImplementedError("ParserState.parse()")

    @staticmethod
    def next(line):
        '''abstract'''
        raise NotImplementedError("ParserState.next()")

class PreParse(ParserState):
    '''Initial ParserState'''

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
    '''handle a parameter'''

    @staticmethod
    def parse(line, this_response):
        '''Parse: param=value'''
        this_response.add_query_param(line)
        return this_response

    @staticmethod
    def next(line):
        if line.is_empty():
            return EmptyItem
        elif line.is_param():
            return ParameterItem
        else:
            raise RuntimeError("Parameter should be followed by another parameter or an empty line")

class EmptyItem(ParserState):
    '''handle an empty line'''

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
            raise RuntimeError("expected either a DATACENTER or another empty line ["
                               + str(line) + "]")

class DatacenterItem(ParserState):
    '''handle data center'''

    @staticmethod
    def parse(line, this_response):
        '''Parse: DATACENTER=id,http://url...'''
        _, rest = str(line).split('=')
        active_id, url = rest.split(',')
        this_response = FederatedResponse(active_id)
        this_response.add_service("DATACENTER", url)
        return this_response

    @staticmethod
    def next(line):
        if line.is_service():
            return ServiceItem
        else:
            raise RuntimeError("DATACENTER line should be followed by a service")

class ServiceItem(ParserState):
    '''handle service description'''

    @staticmethod
    def parse(line, this_response):
        '''Parse: SERICENAME=http://service.url/'''
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
            raise RuntimeError("Service desc. should be followed by a request or another service")

class RequestItem(ParserState):
    '''handle request lines'''

    @staticmethod
    def parse(line, this_response):
        '''Parse: NT STA LC CHA YYYY-MM-DDThh:mm:ss YY-MM-DDThh:mm:ss'''
        this_response.add_request_line(line)
        return this_response

    @staticmethod
    def next(line):
        if line.is_request():
            return RequestItem
        elif line.is_empty():
            return EmptyItem
        else:
            raise RuntimeError("Requests should be followed by another request or an empty line")



# main function
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
'''
    import requests
    url = 'https://service.iris.edu/irisws/fedcatalog/1/'
    r = requests.get(url + "query", params={"net":"A*", "sta":"OK*", "cha":"*HZ"}, verify=False)

    frp = FederatedResponseManager(r.text)
    for n in frp:
        print(n.request("STATIONSERVICE"))
'''