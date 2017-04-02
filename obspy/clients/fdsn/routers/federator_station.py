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
    '''line from federator source that provides additional tests'''
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
            raise RuntimeError("expected either a DATACENTER or another empty line")

class DatacenterItem(ParserState):
    '''handle data center'''

    @staticmethod
    def parse(line, this_response):
        '''Parse: DATACENTER=id,http://url...'''
        _, rest = str(line).split('=')
        active_id, url = rest.split(',')
        this_response = new_federated_response(active_id)
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

def parse_federated_response(block_text):
    '''create a list of FederatedResponse objects, one for each datacenter in response'''
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
    if len(fed_resp > 0) and (not fed_resp[-1].request_lines):
        del fed_resp[-1]
    return fed_resp

class StreamingFederatedResponseParser(object):
    '''Iterate through stream, returning FederatedResponse objects for each datacenter'''
    def __init__(self, stream_iterator):
        self.stream_iterator = stream_iterator() # stream_iterator feeds line by line
        self.state = PreParse
        self.datacenter = FederatedResponse("PRE_CENTER")
        self.parameters = None
        self.line = None

    def __iter__(self):
        return self

    def next(self):
        request_was_processed = False
        if self.line is not None:
            self.datacenter = self.state.parse(self.line, self.datacenter)
            self.line = None

        for self.line in self.stream_iterator:
            self.line = RequestLine(self.line)
            self.state = self.state.next(self.line)
            if request_was_processed and (self.state is not RequestItem):
                self.datacenter.parameters = self.parameters
                return self.datacenter
            if self.state == DatacenterItem and self.datacenter.code == "PRE_CENTER":
                self.parameters = self.datacenter.parameters
            self.datacenter = self.state.parse(self.line, self.datacenter)
            if self.state == RequestItem:
                request_was_processed = True
        raise StopIteration

    __next__ = next

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

    pass_through_params = {
        "DATASELECTSERVICE":["longestonly", "quality", "minimumlength"],
        "STATIONSERVICE":["level", "matchtimeseries", "includeavailability",
                          "includerestricted", "format"]}

    def __init__(self, code):
        self.code = code
        self.parameters = []
        self.services = {}
        self.request_lines = []

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

    def request_text(self, target_service):
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

    def __repr__(self):
        return self.code + "\n" + self.request_text("STATIONSERVICE")

''' inv2x_set(inv) will be used to quickly decide what exists and what doesn't'''
def inv2channel_set(inv):
    A = set()
    for n in inv:
        for s in n:
            for c in s:
                A.add(n.code + "." + s.code + "." + c.location_code + "." + c.code)
    return A

def inv2station_set(inv):
    A = set()
    for n in inv:
        for s in n:
            A.add(n.code + "." + s.code)
    return A

def inv2network_set(inv):
    A = set()
    for n in inv:
        A.add(n.code)
    return A

def req2network_set(req):
    A = set()
    for line in req:
        (net,sta,loc,cha,startt,endt) = r.split()
        A.add(net)

def req2station_set(req):
    A = set()
    for line in req:
        (net,sta,loc,cha,startt,endt) = r.split()
        A.add(net + "." + sta)

def req2channel_set(req):
    A = set()
    for line in req:
        (net,sta,loc,cha,startt,endt) = r.split()
        A.add(net + "." + sta + "." + loc + "." + cha)

''' converters used to make comparisons between inventory items and requests '''
req_converter = {"channel":req2channel_set, "station":req2station_set, "network":req2network_set}
inv_converter = {"channel":inv2channel_set, "station":inv2station_set, "network":inv2network_set}

def request_exists_in_inventory(inv, requests, level):
    # does not account for timespans, only for existence net-sta-loc-cha matches

    inv_set = inv_converter[level](inv)
    req_set = req_converter[level](requests)
    members = req_set.intersection(inv_set)
    non_members = req_set.difference(inv_set)
    return members, non_members

def fed_get_stations(**kwarg):
    '''This will be the request for the federated station service'''

    def add_datacenter_reference(inventory_list, code, service_urls):
        '''Add a tag to each inventory item (what level?) attributing to a datacenter
        '''
        #TODO as per:https://docs.obspy.org/tutorial/code_snippets/quakeml_custom_tags.html
        # Tried, but this doesn't seem to appear in the final xml
        extra = {
            'code': {'value': code,
                              'namespace': r"http://www.fdsn.org/xml/station/"},
            'datacenter_url': {'value': service_urls["DATACENTER"],
                               'namespace': r"http://www.fdsn.org/xml/station/"},
            'service_url': {'value': service_urls["STATIONSERVICE"],
                            'namespace': r"http://www.fdsn.org/xml/station/"}}

        inv.extra = extra #tags don't seem to work
        for x in inv:
            x.extra = extra

    LEVEL = kwarg["level"]
    remap = {"IRISDMC":"IRIS", "GEOFON":"GFZ", "SED":"ETH", "USPSC":"USP"}
    # need to add to URL_MAPPINGS!
    all_inv = []
    lines_to_resubmit = []
    succesful_retrieves = []

    url = 'https://service.iris.edu/irisws/fedcatalog/1/'
    kwarg["targetservice"]="STATION"
    r = requests.get(url + "query", params=kwarg, verify=False)
    print "asking from..."
    for p in r.iter_lines():
        if p.startswith("DATACENTER"):
            print(p)

    sfrp = StreamingFederatedResponseParser(r.iter_lines)
    for datac in sfrp:
        dc_id = datac.code

        if dc_id in remap:
            dc_id = remap[dc_id]
        client = Client(dc_id)

        #TODO large requests could be denied (code #413) and will need to be chunked.
        print(datac.request_text("STATIONSERVICE").count('\n'))
        #TODO datac.request_txt could be vetted ahead of time here by comparing to what we have. more likely during 2nd round
        try:
            inv = client.get_stations_bulk(bulk=datac.request_text("STATIONSERVICE"))
        except: #except expression as identifier:
            lines_to_resubmit.extend(datac.request_lines)
            print(dc_id, "error!")
        else:
            successful, failed = request_exists_in_inventory(inv, datac.request_lines, LEVEL)
            successful_retrieves = datac.request_lines
            add_datacenter_reference(inv, datac.code, datac.services)
            if not all_inv:
                all_inv = inv
                all_inv_set = inv_converter[LEVEL](inv)
            else:
                all_inv += inv
                all_inv_set= all_inv_set.union(inv_converter[LEVEL](inv)
    return all_inv


# main function
if __name__ == '__main__':
    #import doctest
    #doctest.testmod(exclude_empty=True)

    import requests
    url = 'https://service.iris.edu/irisws/fedcatalog/1/'
    r = requests.get(url + "query", params={"net":"A*", "sta":"OK*", "cha":"*HZ"}, verify=False)

    frp = StreamingFederatedResponseParser(r.iter_lines)
    for n in frp:
        print(n.request_text("STATIONSERVICE"))
