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

from obspy.clients.fdsn.routers import FederatedResponseParser, StreamingFederatedResponseParser, FederatedResponse

''' inv2x_set(inv) will be used to quickly decide what exists and what doesn't'''
def inv2channel_set(inv):
    return {n.code + "." + s.code + "." +
            c.location_code + "." + c.code
            for n in inv for s in n for c in s}

def inv2station_set(inv):
    return {n.code + "."+ s.code for n in inv for s in n}

def inv2network_set(inv):
    return {n.code for n in inv}

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
            x.network.extra = extra

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
    print([p for p in r.iter_lines() if p.startswith("DATACENTER")])

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
            lines_to_resubmit.extend(datac.request_lines) #unsuccessful attempt. Add all requests into resubmit queue
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
                all_inv_set = all_inv_set.union(inv_converter[LEVEL](inv))

    # now, perhaps we got lucky, and have all the data we requested?
    if not failed:
        return all_inf
    
    # okey-dokey. Time for round 2. # # #
    # resubmit the failed retrieves to the federator service
    
    # as data is retrieved, add to all_inv and remove it from the queue.



    return all_inv


# main function
if __name__ == '__main__':
    #import doctest
    #doctest.testmod(exclude_empty=True)

    import requests
    URL = 'https://service.iris.edu/irisws/fedcatalog/1/'
    R = requests.get(URL + "query", params={"net":"A*", "sta":"OK*", "cha":"*HZ"}, verify=False)

    FRP = StreamingFederatedResponseParser(R.iter_lines)
    for n in FRP:
        print(n.request_text("STATIONSERVICE"))
