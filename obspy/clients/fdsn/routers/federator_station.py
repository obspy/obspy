from obspy.clients.fdsn import Client
'''
from obspy import UTCDateTime
starttime = UTCDateTime("2001-01-01")

# to access metadata via Client from a single data center's station service
from obspy.clients.fdsn import Client
client = Client("IRIS")

# or, access data via FederatorClient from multiple data centers
from obspy.clients.fdsn import FederatorClient
client = FederatorClient("IRIS")


# the requests are submited in exactly the same manner
inv = client.get_stations(network="IU", station="A*", starttime=starttime)
'''


'''
service-options 	:: service specific options
targetservice=<station|dataselect>
format=<request|text>
includeoverlaps=<true|false>

active-options 	 	::
[channel-options] [map-constraints] [time-constraints]
  channel-options 	::
  net=<network>
  sta=<station>
  loc=<location>
  cha=<channel>

  map-constraints 	::  ONE OF
     boundaries-rect 	::
        minlatitude=<degrees>
        maxlatitude=<degrees>
        minlongitude=<degrees>
        maxlongitude=<degrees>
     boundaries-circ 	::
        latitude=<degrees>
        longitude=<degrees>
        maxradius=<number>
        minradius=<number>

  time-constraints 	::
    starttime=<date>
    endtime=<date>
    startbefore=<date>
    startafter=<date>
    endbefore=<date>
    endafter=<date>
    updatedafter=<date>]

passive-options 	::
    includerestricted=<true|false>
    includeavailability=<true|false>
    matchtimeseries=<true|false>
    longestonly=<true|false>
    quality=<D|R|Q|M|B>
    level=<net|sta|cha|resp>
    minimumlength=<number>
'''

class RequestLine(object):
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
    parser states: PREPARSE, PARAMLIST, EMPTY_LINE, DATACENTER, SERVICE, REQUEST, DONE
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
        raise NotImplementedError, "ParserState.parse()"

    @staticmethod
    def next(line):
        '''abstract'''
        raise NotImplementedError, "ParserState.next()"

class PreParse(ParserState):
    '''Initial ParserState'''

    @staticmethod
    def parse(line, this_response):
        return this_response

    @staticmethod
    def next(line):
        if line.is_empty():
            return EmptyItem #EMPTY_LINE
        elif line.is_datacenter():
            return DatacenterItem #DATACENTER
        elif line.is_param():
            return ParameterItem #PARAMLIST
        else:
            return ParserState

class ParameterItem(ParserState):
    '''handle a parameter'''

    @staticmethod
    def parse(line, this_response):
        '''Parse: param=value'''
        this_response.add_common_parameters(line)
        return this_response

    def next(self, line):
        if line.is_empty():
            return EmptyItem() #EMPTY_LINE
        elif line.is_param():
            return self
        else:
            raise RuntimeError, "Parameter should be followed by another parameter or an empty line"

class EmptyItem(ParserState):
    '''handle an empty line'''

    @staticmethod
    def parse(line, this_response):
        return this_response
    
    @staticmethod
    def next(line):
        if line.is_empty():
            return EmptyItem #no state change
        elif line.is_datacenter():
            return DatacenterItem #DATACENTER
        else:
            raise RuntimeError, "expected either a DATACENTER or another empty line"

class DatacenterItem(ParserState):
    '''handle data center'''

    @staticmethod
    def parse(line, this_response):
        '''Parse: DATACENTER=id,http://url...'''
        _, rest = str(line).split('=')
        active_id, url = rest.split(',')
        this_response = FederatedResponseParser.new_federated_response(active_id)
        print("new response", this_response)
        this_response.add_service("DATACENTER", url)
        return this_response

    @staticmethod
    def next(line):
        if line.is_service():
            return ServiceItem
        else:
            raise RuntimeError, "DATACENTER line should be followed by a service"

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
            raise RuntimeError, "Service desc. should be followed by a request or another service"

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
            raise RuntimeError, "Requests should be followed by another request or an empty line"

class FederatedResponseParser(object):
    '''Iterate through stream, returning FederatedResponse objects for each datacenter'''
    def __init__(self, stream_iterator):
        self.stream_iterator = stream_iterator() # stream_iterator feeds us line by line
        self.state = PreParse
        self.n_datacenters = 0
        self.fed_req = None
        self.line = None

    def __iter__(self):
        return self
        
    def next(self):
        request_was_processed = False
        print("A Next...", self.state)
        if self.line is not None:
            print("returned to NEXT")
            self.fed_req = self.state.parse(self.line, self.fed_req) #left before processing
            self.line = None
        print(self.state)
        for self.line in self.stream_iterator:
            self.line = RequestLine(self.line)
            self.state = self.state.next(self.line)
            print(self.state)
            if request_was_processed and (self.state is not RequestItem):
                return self.fed_req
            self.fed_req = self.state.parse(self.line, self.fed_req)
            if self.state == RequestItem:
                request_was_processed = True
        raise StopIteration
            
    __next__ = next

    @staticmethod
    def new_federated_response(ds_id):
        return FederatedResponse(ds_id)

    @staticmethod
    def new_federated_response(ds_id):
        return FederatedResponse(ds_id)

class FederatedResponse(object):
    '''
    >>> fed_resp = FederatedResponse("IRISDMC")
    >>> fed_resp.add_common_parameters(["lat=50","lon=20","level=cha"])
    >>> fed_resp.add_service("STATIONSERVICE","http://service.iris.edu/fdsnws/station/1/")
    >>> fed_resp.add_request_line("AI ORCD -- BHZ 2015-01-01T00:00:00 2016-01-02T00:00:00")
    >>> fed_resp.add_request_line("AI ORCD 04 BHZ 2015-01-01T00:00:00 2016-01-02T00:00:00")
    >>> print(fed_resp.request_text("STATIONSERVICE"))

    level=cha
    AI ORCD -- BHZ 2015-01-01T00:00:00 2016-01-02T00:00:00
    AI ORCD 04 BHZ 2015-01-01T00:00:00 2016-01-02T00:00:00
    '''

    ok_parameters = {"DATASELECTSERVICE":["longestonly"],
                        "STATIONSERVICE":["level"]}

    def __init__(self, datacenter_id):
        self.datacenter_id = datacenter_id
        self.common_parameters = []
        self.services = {}
        self.request_lines = []

    def add_service(self, service_name, service_url):
        self.services[service_name] = service_url

    def add_common_parameters(self, common_parameters):
        if isinstance(common_parameters, str):
            self.common_parameters.append(common_parameters)
        elif isinstance(common_parameters, RequestLine):
            self.request_lines.append(str(common_parameters))
        else:
            self.common_parameters.extend(common_parameters)

    def add_request_lines(self, request_lines):
        if isinstance(request_lines, str):
            self.request_lines.append(request_lines)
        elif isinstance(request_lines, RequestLine):
            self.request_lines.append(str(request_lines))
        else:
            self.request_lines.extend(request_lines)

    def add_request_line(self, request_line):
        self.request_lines.append(request_line)

    def request_text(self, target_service):
        reply = self.selected_common_parameters(target_service)
        reply.extend(self.request_lines)
        return "\n".join(reply)

    def selected_common_parameters(self, target_service):
        reply = []
        for good in FederatedResponse.ok_parameters[target_service]:
            reply.extend([c for c in self.common_parameters if c.startswith(good + "=")])
        return reply
    
    def __repr__(self):
        return self.datacenter_id + "\n" + self.request_text("STATIONSERVICE")
        
# main function
if __name__ == '__main__':
    #import doctest
    #doctest.testmod(exclude_empty=True)

    import requests
    url='https://service.iris.edu/irisws/fedcatalog/1/'
    r=requests.get(url + "query", params={"net":"A*","sta":"OK*","cha":"*HZ"}, verify=False)

    frp = FederatedResponseParser(r.iter_lines)
    for n in frp:
        print(n.request_text("STATIONSERVICE"))
