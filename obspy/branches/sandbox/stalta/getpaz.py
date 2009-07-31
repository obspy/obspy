#!/usr/bin/env python
# REQUIRES PYTHON 2.5

from obspy.seishub.client import Client
from pprint import pprint

client = Client("http://admin:admin@teide.geophysik.uni-muenchen.de:8080")

def getStationPAZ(network_id, station_id, datetime):
    """
    Get PAZ for Station to certain time

    @parm network_id: Network id, e.g. BW
    @parm station_id: Station id, e.g. RJOB
    @parm datetime: Time, e.g. 20090707 or 20090707200000
    @return: Dictionary containing zeroes, poles and gain

    >>> getStationPAZ('BW','RJOB','20090707')
    ({'zeroes': [0j, 0j], 'poles': [(-0.037004000000000002+0.037016j), \
(-0.037004000000000002-0.037016j), (-251.33000000000001+0j), \
(-131.03999999999999-467.29000000000002j), \
(-131.03999999999999+467.29000000000002j)], 'gain': 60077000.0}, \
2516800000.0)
    """
    # request station information
    station_list = client.station.getList(network_id=network_id, 
                                          station_id=station_id,
                                          datetime=datetime)
    if len(station_list)!=1:
        return {}
    # request station resource
    res = client.station.getXMLResource(station_list[0]['resource_name'])
    # get poles
    node = res.station_control_header.response_poles_and_zeros
    poles_real = node.complex_pole.real_pole[:]    
    poles_imag = node.complex_pole.imaginary_pole[:]
    poles = zip(poles_real, poles_imag)
    poles = [p[0] + p[1]*1j for p in poles]
    # get zeros
    node = res.station_control_header.response_poles_and_zeros
    zeros_real = node.complex_zero.real_zero[:][:]    
    zeros_imag = node.complex_zero.imaginary_zero[:][:]
    zeros = zip(zeros_real, zeros_imag)
    zeros = [p[0] + p[1]*1j for p in zeros]
    # get gain and sensitivity
    node = res.station_control_header
    gains = node.xpath('response_poles_and_zeros/A0_normalization_factor')[-1]
    gain = gains[-1]
    sensitivities = node.xpath('channel_sensitivity_gain/sensitivity_gain')
    sensitivity = sensitivities[-1]
    return {'poles': poles, 'zeroes': zeros, 'gain': gain}, sensitivity

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
