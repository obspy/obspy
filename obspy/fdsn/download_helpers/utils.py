from collections import OrderedDict
from obspy.core.util import locations2degrees

def filename_constructor(net, sta, loc, cha, 
        pattern='{net}.{sta}.{loc}.{cha}'):
    '''
    Construct arbitrary file names for saving the data
    filename = filename_constructor('iN', 'iS', 'iL', 'iC', 
                pattern='MYFORMAT.{sta}.{net}.{cha}.{loc}.EXTENTION')
    print filename
    'MYFORMAT.iS.iN.iC.iL.EXTENTION'
    '''
    return pattern.format(net=net, sta=sta, loc=loc, cha=cha)

def filter_availability(dict_client1, dict_client2, 
                            dist_threshold=20):
    '''
    Filter one list of stations with another one!
    dict_client1 = \
    {"network.station": {
         "latitude": 1.0,
         "longitude": 2.0,
         "elevation_in_m": 10.0,
         "channels": [".BHE", ".BHN", ".BHZ", "00.LHE", "00.LHE", ...]},
      "network.station": {...}, ...
     }
    '''
    dict_client1 = OrderedDict(sorted(
        dict_client1.items(), key=lambda x: x[1]["latitude"]))
    dict_client2 = OrderedDict(sorted(
        dict_client2.items(), key=lambda x: x[1]["latitude"]))

    filtered_client2 = OrderedDict()
    for key, value in dict_client2.items():
        if key in dict_client1:
            continue
        filtered_client2[key] = value
    dict_client2 = filtered_client2

    lat_thresh = 1.0
    for key2, info2 in dict_client2.items():
        print key2
        for key1, info1 in dict_client1.items():
            if abs(info1['latitude']-info2['latitude']) < lat_thresh:
                dist12 = locations2degrees(info1['latitude'], info1['longitude'],
                                           info2['latitude'], info2['longitude'])
                dist12_m = dist12*111.*1000.
                print dist12_m
                if dist12_m < dist_threshold:
                    print 'Within the vicinity!'
                    break

            if info1['latitude'] > info2['latitude']:
                # no way to come back...!
                dict_client1[key2] = info2
                break
            else:
                # Maybe the next station?
                if (dict_client1.keys().index(key1) == 
                        len(dict_client1)-1):
                    dict_client1[key2] = info2
                continue
