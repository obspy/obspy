from __future__ import print_function
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
import requests
import sys
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
    inv = client.get_stations(network="IU", station="req_str*", starttime=starttime)
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

from federator_response_parser import parse_federated_response, FederatedResponse, filter_requests
#from obspy.clients.fdsn.routers import parse_federated_response, FederatedResponse

# inv2x_set(inv) will be used to quickly decide what exists and what doesn't
def inv2channel_set(inv):
    'return a set containing string representations of an inv object'
    return {n.code + "." + s.code + "." +
            c.location_code + "." + c.code
            for n in inv for s in n for c in s}

def inv2station_set(inv):
    'return a set containing string representations of an inv object'
    return {n.code + "."+ s.code for n in inv for s in n}

def inv2network_set(inv):
    'return a set containing string representations of an inv object'
    return {n.code for n in inv}

def req2network_set(req):
    'return a set containing string representations of an request line'
    req_str = set()
    for line in req:
        (net, sta, loc, cha, startt, endt) = line.split()
        req_str.add(net)
    return req_str

def req2station_set(req):
    'return a set containing string representations of an request line'
    req_str = set()
    for line in req:
        (net, sta, loc, cha, startt, endt) = line.split()
        req_str.add(net + "." + sta)
    return req_str

def req2channel_set(req):
    'return a set containing string representations of an request line'
    req_str = set()
    for line in req:
        (net, sta, loc, cha, startt, endt) = line.split()
        req_str.add(net + "." + sta + "." + loc + "." + cha)
    return req_str

#converters used to make comparisons between inventory items and requests
INV_CONVERTER = {"channel":inv2channel_set, "station":inv2station_set, "network":inv2network_set}
REQ_CONVERTER = {"channel":req2channel_set, "station":req2station_set, "network":req2network_set}

def request_exists_in_inventory(inv, requests, level):
    'compare inventory to requests'
    # does not account for timespans, only for existence net-sta-loc-cha matches
    inv_set = INV_CONVERTER[level](inv)
    req_set = REQ_CONVERTER[level](requests)
    members = req_set.intersection(inv_set)
    non_members = req_set.difference(inv_set)
    return members, non_members

REMAP = {"IRISDMC":"IRIS", "GEOFON":"GFZ", "SED":"ETH", "USPSC":"USP"}

def get_stations(exclude_datacenter=None, include_datacenter=None, includeoverlaps=False, **kwarg):
    '''This will be the original request for the federated station service
    :param exclude_datacenter: Avoids getting data from datacenters with the specified ID code(s)
    :param include_datacenter: limit retrieved data to  datacenters with the specified ID code(s)
    :param includeoverlaps: For now, simply leave this False. It will confuse the program.
    other parameters as seen in Client.get_stations

    Warning: If you specify "include_datacenter", you may miss data that the datacenter holds, but isn't the primary source
    Warning: If you specify "exclude_datacenter", you may miss data for which that datacenter is the primary source

    '''

    # peel away arguments that are specific to the federator
    # include/exclude works with the OBSPY set of codes (remapped) eg. IRIS vs IRISDMC

    level = kwarg["level"]

    all_inv = []
    lines_to_resubmit = []
    succesful_retrieves = []

    # send request to the FedCatalog
    url = 'https://service.iris.edu/irisws/fedcatalog/1/'
    params = kwarg.copy()
    params["targetservice"] = "STATION"
    resp = requests.get(url + "query", params=params, verify=False)

    sfrp = parse_federated_response(resp.text)

    # remap datacenter codes because IRIS codes differ from OBSPY codes
    for dc in sfrp:
        if dc.code in REMAP:
            dc.code = REMAP[dc.code]

    sfrp = filter_requests(sfrp, include_datacenter=include_datacenter,
                           exclude_datacenter=exclude_datacenter)

    # get data from each datacenter
    for datac in sfrp:
        code = datac.code
        print("requesting data from:" + code + " : " + datac.services["DATACENTER"])
        try:
            client = Client(code)
        except Exception as ex:
            print("Problem assigning client " + code, file=sys.stderr)
            print (ex, __type__(ex), ex.__class__, file=sys.stderr)
            lines_to_resubmit.extend(datac.request_lines)
            raise
            continue

        #TODO large requests could be denied (code #413) and will need to be chunked.
        print("number of items:" + str(datac.request_text("STATIONSERVICE").count('\n')))
        try:
            inv = client.get_stations_bulk(bulk=datac.request_text("STATIONSERVICE"))
        except FDSNNoDataException as ex:
            lines_to_resubmit.extend(datac.request_lines)
            print("no data available")
            print(ex, file=sys.stderr)
        except Exception as ex: #except expression as identifier:
            lines_to_resubmit.extend(datac.request_lines)
            print(code + " error!", file=sys.stderr)
            print(ex, file=sys.stderr)
            raise
            continue
        # try to figure out if we got what we came for
        successful, failed = request_exists_in_inventory(inv, datac.request_lines, level)

        if not all_inv:
            all_inv = inv
            all_inv_set = INV_CONVERTER[level](inv)
        else:
            all_inv += inv
            all_inv_set = all_inv_set.union(INV_CONVERTER[level](inv))
        print("done with " + code)
    # now, perhaps we got lucky, and have all the data we requested?
    if not lines_to_resubmit:
        print("It appears we may have [possibly [maybe]] all requested data")
        return all_inv
    else:
        print("The following requests were unfulfilled:")
        print([dc.request_lines for dc in failed])

    print("Here is where we *would* resumit the following to federator service")
    print("\n".join([line for line in lines_to_resubmit]))
    return all_inv

if __name__ == '__main__':
    #import doctest
    #doctest.testmod(exclude_empty=True)
    INV = get_stations(network="A*", station="OK*", channel="*HZ", level="station")
    print(INV)
    INV = get_stations(network="*", station="AN*", level="station")
    print(INV)
    '''
    import requests
    URL = 'https://service.iris.edu/irisws/fedcatalog/1/'
    RQST = requests.get(URL + "query", params={"net":"A*", "sta":"OK*", "cha":"*HZ"}, verify=False)

    FRP = parse_federated_response(RQST.text)
    for n in FRP:
        print(n.request_text("STATIONSERVICE"))
        direct_client = Client('iris')
        direct_inv = direct_client.get_stations_bulk(n.request_text("STATIONSERVICE"))
        print(direct_inv)
        #fed_get_stations()
        '''
