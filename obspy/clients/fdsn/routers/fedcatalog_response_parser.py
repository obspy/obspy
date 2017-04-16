#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
from obspy.clients.fdsn import Client
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
        return self.line == ""

    def is_datacenter(self):
        return self.line.startswith('DATACENTER=')

    def is_param(self):
        # true for datacenter, services, and parameter_list
        return '=' in self.line

    def is_request(self):
        return len(self.line.split()) == 6 # and test field values?

    def is_service(self):
        # parse param_name
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
        this_response.add_common_parameters(line)
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
            raise RuntimeError("expected either a DATACENTER or another empty line ["+ str(line)+"]")

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
        this_response.add_request_lines(line)
        return this_response

    @staticmethod
    def next(line):
        if line.is_request():
            return RequestItem
        elif line.is_empty():
            return EmptyItem
        else:
            raise RuntimeError("Requests should be followed by another request or an empty line")



class FederatedResponse(object):
    '''
    >>> fed_resp = FederatedResponse("IRISDMC")
    >>> fed_resp.add_common_parameters(["lat=50","lon=20","level=cha"])
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
        self.code = code
        self.parameters = []
        self.services = {}
        self.request_lines = []

    def __len__(self):
        return len(self.request_lines)

    def add_service(self, service_name, service_url):
        self.services[service_name] = service_url

    def add_common_parameters(self, parameters):
        if isinstance(parameters, str):
            self.parameters.append(parameters)
        elif isinstance(parameters, RequestLine):
            self.parameters.append(str(parameters))
        else:
            self.parameters.extend(parameters)

    def add_request_lines(self, request_lines):
        '''append one or more requests to the request list'''
        if isinstance(request_lines, str):
            self.request_lines.append(request_lines)
        elif isinstance(request_lines, RequestLine):
            self.request_lines.append(str(request_lines))
        else:
            self.request_lines.extend(request_lines)

    def add_request_line(self, request_line):
        '''append a single request to the list of requests'''
        self.request_lines.append(request_line)

    def text(self, target_service):
        '''Return a string suitable for posting to a target service'''
        reply = self.selected_common_parameters(target_service)
        reply.extend(self.request_lines)
        return "\n".join(reply)

    def selected_common_parameters(self, target_service):
        '''Return common parameters, targeted for a specific service
        This effecively filters out parameters that don't belong in a request.
        for example, STATIONSERVICE can accept level=xxx ,
        while DATASELECTSERVICE can accept longestonly=xxx
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

    def __repr__(self):
        return self.code + "\n" + self.text("STATIONSERVICE")

    def client(self, **kwarg):
        '''returns appropriate obspy.clients.fdsn.clients.Client for request'''
        try:
            client = Client(self.code, kwarg)
        except Exception as ex:
            print("Problem assigning client " + code, file=sys.stderr)
            print (ex, __type__(ex), ex.__class__, file=sys.stderr)
            raise



# main function
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
'''
    import requests
    url = 'https://service.iris.edu/irisws/fedcatalog/1/'
    r = requests.get(url + "query", params={"net":"A*", "sta":"OK*", "cha":"*HZ"}, verify=False)

    frp = parse_federated_response(r.text)
    for n in frp:
        print(n.request("STATIONSERVICE"))
'''