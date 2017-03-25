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


'''
station tests:

https://service.iris.edu/irisws/fedcatalog/1/query?sta=A*&minlat=34&maxlat=38&cha=?HZ&starttime=2015-01-01&includeoverlaps=true
gives:


'''



client = Client("iris")

def get_station(parameter_list):
    pass

class ResponseManager(object):
    ERROR, PREPARSE, PARAMLIST, EMPTY_LINE, DATACENTER, SERVICE, REQUEST, DONE = range(7)

    # the parameters at the top of the request file
    common_parameters = {}

    # dict: datacenter_urls[DATACENTER_CODE]=base_url
    datacenter_urls = {}
    # dict: services[SERVICE_NAME]
    services = {}
    def isEmpty(self, line):
        return line.isempty()
    
    def isDatacenter(self, line):
        return self.current_line.startswith('DATACENTER=')

    def isParam(self):
        # true for datacenter, services, and parameter_list
        return self.current_line.has("=")

    def isRequest(self):
        return self.current_line.nfields()==6 # and test field values?

    def isService(self):
        # parse param_name
        return self.current_line.isParam(line) and allcapsParamName(param_name); 

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

        def parse(self) :
            assert 0, "undefined state"
            pass

        def next(self, input):
            pass

    class PreParse(State):
        def parse(self) :
            pass

        def next(self, line):
            if line.isEmpty():
                return EmptyItem #EMPTY_LINE
            elif line.isDatacenter():
                return DatacenterItem #DATACENTER
            elif line.isParam():
                return ParameterItem #PARAMLIST
            else:
                return State

    class ParameterItem(State):
        def parse(self) :
            # add line to PARAMLIST dict
            pass

        def next(self, item):
            if line.isEmpty():
                return EmptyItem #EMPTY_LINE
            elif line.isParam():
                return self
            else:
                assert 0, "expected another paramter or an empty line"
                return State

    class EmptyItem(State):
        def parse(self) :
            pass

        def next(self, item):
            if line.isEmpty():
                return self #no state change
            elif line.isDatacenter():
                return DatacenterItem #DATACENTER
            else:
                assert 0, "expected either a datacenter or an empty line"
                return State

    class DatacenterItem(State):
        def parse(self) :
            # set new datacenter in dictionary. Example:
            '''DATACENTER=IRISDMC,http://ds.iris.edu'''
            # current key becomes IRISDMC
            # x[IRISDMC].url = http://ds.iris.edu
            pass

        def next(self, item):
            if line.isService():
                return ServiceItem
            else:
                assert 0, "expected a service" 
                return State

    class ServiceItem(State):
        def parse(self) :
            # add service to dictionary. example
            '''DATASELECTSERVICE=http://service.iris.edu/fdsnws/dataselect/1/'''
            # x[IRISDMC].DATASELECT = http://service.iris.edu/fdsnws/dataselect/1/
            pass

        def next(self, item):
            if line.isService():
                return self
            elif line.isRequest():
                return RequestItem
            else:
                assert 0, "expected either a request or another service"

    class RequestItem(State):
        def parse(self) :
            # add request to this service's requests. example:
            # x[IRISDMC].request.add('AR AJAR -- EHZ 2015-01-01T00:00:00 2016-01-02T00:00:00')
            pass

        def next(self, item):
            if line.isRequest():
                return self
            elif line.isEmpty():
                return EmptyItem
            else:
                assert 0, "expected either another request or an empty line"


    def __init__(self, initial_state):
        pass

    def parse(full_federator_response):
        current_line = ""
        next_line = ""


        '''
        first line of each station will be DATACENTER=VAL,http://stuff.place
        For example:

        DATACENTER=IRISDMC,http://ds.iris.edu
        DATASELECTSERVICE=http://service.iris.edu/fdsnws/dataselect/1/
        STATIONSERVICE=http://service.iris.edu/fdsnws/station/1/
        EVENTSERVICE=http://service.iris.edu/fdsnws/event/1/
        SACPZSERVICE=http://service.iris.edu/irisws/sacpz/1/
        RESPSERVICE=http://service.iris.edu/irisws/resp/1/
        AR AJAR -- EHZ 2015-01-01T00:00:00 2016-01-02T00:00:00

        File format:
        [BLOCK WITH 1 OR MORE param=value PAIRS]
        [blank line]
        [LINE WITH DATACENTER= ]
        [BLOCK WITH 1 OR MORE SERVICES  SVCNAME=http://something.../v/]
        [BLOCK OF REQUESTS]
        [blank line]
        [LINE WITH DATACENTER= ]
        [BLOCK WITH 1 OR MORE SERVICES  SVCNAME=http://something.../v/]
        [BLOCK OF REQUESTS]

        Interesting bits for parsing: 
        1. first block may or may not exist [is it necessary?  some parts of it would be.]
        2. each section starts with an empty line, even the first one of the list [meh-reliable]
        3. every service ends in "/" [more reliable]
        4. data lines always have 6 fields
        5. spaces only exist between fields, and are confined to the data lines.
        6. any number of blank lines might trail the file
        '''



        state = PreParse;
        for line in file:
            state = state.next(line)
            state.run()

        FedResponses = [FederatorResponse(s) for s in datacenter_dict]


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
