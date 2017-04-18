from obspy.clients.fdsn import FederatedClient, Client
import requests
# DO NOT USE THIS.. IT IS A TEMPORARY FILE HOLDING IDEAS ONLY
def get_station_bulk(object):
    '''Retrieve station data via fedcatalog'''
    client = FederatedClient("IRIS")
    url = 'https://service.iris.edu/irisws/fedcatalog/1/'
    data = 'includeoverlaps=true\nA* OR* * BHZ 2015-01-01T00:00:00 2016-01-02T00:00:00'
    resp = requests.post(url + "query", data=data, verify=False, stream=True)
    if not resp.ok:
        #deal with it
        pass
    if resp.ok:
        # parse each line
        if save_to_file:
            with open(filename, 'wb') as fd:
                for chunk in resp.content(chunk_size=128):
                    fd.write(chunk)
        else:
            for line in resp_iter_lines():
                # process line
                pass

    # now, try again with includeoverlaps
