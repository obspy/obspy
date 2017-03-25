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


class DatacenterRequestDetails(object):
    def __init__(self, ID):
        self.ID = ID
        self.url={}
        self.request=[]

    def add_url(self, service, url):
        self.url[service]=url

    def add_request(self, request_line):
        self.request.append(request_line) 
        
    def __str__(self):
        ss = str(self.ID) + "\n"
        for ur in self.url.keys():
            ss += ur + " at " + self.url[ur] +'\n'
        for r in self.request:
            ss += str(r) + '\n'
        return ss


class request_line(object):
    def isEmpty(self):
        return self.line == ""
    
    def isDatacenter(self):
        return self.line.startswith('DATACENTER=')

    def isParam(self):
        # true for datacenter, services, and parameter_list
        return '=' in self.line

    def isRequest(self):
        return len(self.line.split())==6 # and test field values?

    def isService(self):
        # parse param_name
        return self.isParam() and self.line.split("=")[0].isupper() and not self.isDatacenter()

        
    def __init__(self, line):
        self.line = line.strip()
    
    def __repr__(self):
        return self.line
    
    def __str__(self):
        return self.line

class ResponseManager:
    common_parameters=[]
    datacenter_ids = [] #strings
    datacenters={} # DatacenterRequestDetails
    active_id=""
    def __str__(self):
        ss = str(ResponseManager.common_parameters)
        ss += str(ResponseManager.datacenter_ids)
        for id in ResponseManager.datacenter_ids:
            ss += str(ResponseManager.datacenters[id])
            ss +='\n'
        return ss

    def parse(self, full_federator_response):
        state = PreParse();

        for line in full_federator_response.splitlines():
            subject = request_line(line)
            state = state.next(subject)
            state.parse(subject)

        # now, we have:
        #  datacenter_ids : list of datacenter codes, in order received
        #  datacenters : dictionary of DatacenterRequestDetails, by id
        FedResponses = [FederatorResponse(dc, common_parameters) for dc in datacenters]

class State:
    '''
    parser states: PREPARSE, PARAMLIST, EMPTY_LINE, DATACENTER, SERVICE, REQUEST, DONE
    PREPARSE -> [PARAMLIST | EMPTY_LINE | DATACENTER]
    PARAMLIST -> [PARAMLIST | EMPTY_LINE]
    EMPTY_LINE -> [EMPTY_LINE | DATACENTER | DONE]
    DATACENTER -> [SERVICE]
    SERVICE -> [SERVICE | REQUEST]
    REQUEST -> [REQUEST | EMPTY_LINE | DONE ]
    '''

    def parse(self, line) :
        assert 0, "undefined state"
        pass

    def next(self, line):
        pass

class PreParse(State):
    def parse(self, line) :
        pass

    def next(self, line):
        if line.isEmpty():
            return EmptyItem() #EMPTY_LINE
        elif line.isDatacenter():
            return DatacenterItem() #DATACENTER
        elif line.isParam():
            return ParameterItem() #PARAMLIST
        else:
            return State

class ParameterItem(State):
    def parse(self, line) :
        ResponseManager.common_parameters.append(line)

    def next(self, line):
        if line.isEmpty():
            return EmptyItem() #EMPTY_LINE
        elif line.isParam():
            return self
        else:
            assert 0, "expected another paramter or an empty line"
            return State()

class EmptyItem(State):
    def parse(self, line) :
        pass

    def next(self, line):
        if line.isEmpty():
            return self #no state change
        elif line.isDatacenter():
            return DatacenterItem() #DATACENTER
        else:
            assert 0, "expected either a datacenter or an empty line"
            return State()

class DatacenterItem(State):
    def parse(self, line) :
        _, rest =  str(line).split('=')
        active_id, url = rest.split(',')
        ResponseManager.active_id = active_id
        ResponseManager.datacenter_ids.append(active_id)
        ResponseManager.datacenters[active_id]=DatacenterRequestDetails(active_id)
        ResponseManager.datacenters[active_id].add_url(active_id,url)

    def next(self, line):
        if line.isService():
            return ServiceItem()
        else:
            assert 0, "expected a service" 
            return State()

class ServiceItem(State):
    def parse(self, line) :
        '''parsing something like: DATASELECTSERVICE=http://service.iris.edu/fdsnws/dataselect/1/'''
        svc_name, url = str(line).split('=')
        ResponseManager.datacenters[ResponseManager.active_id].add_url(svc_name, url)
        pass

    def next(self, line):
        if line.isService():
            return self
        elif line.isRequest():
            return RequestItem()
        else:
            assert 0, "expected either a request or another service"
            
class RequestItem(State):
    def parse(self, line) :
        # add request to this service's requests. example:
        # x[IRISDMC].request.add('AR AJAR -- EHZ 2015-01-01T00:00:00 2016-01-02T00:00:00')
        ResponseManager.datacenters[ResponseManager.active_id].add_request(line)

    def next(self, line):
        if line.isRequest():
            return self
        elif line.isEmpty():
            return EmptyItem()
        else:
            assert 0, "expected either another request or an empty line"




class FederatorResponse(object):
    def __init__(code, url, params, service_dict, requests):
        self.datacenter_code = code# IGNV, IRIS, etc
        self.datacenter_url = url# http://ds.iris.edu
        self.service_urls = service_dict
        self.parameter_details = params
        self.request_details = requests

    def __str__(self):
        return ( 'Datacenter: {datacenter_code}'

        )
    def request_text(self):
        #put request lines into a good format
        #maybe put subset of params
        r  = "\n".join(str(i) for i in self.request_details)
        p = "\n".join(str(i) for i in self.parameter_details)
        return ('{r}\n{p}').format(r=r, p=p)


# main function
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
