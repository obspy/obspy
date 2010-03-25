# -*- coding: utf-8 -*-
"""
SeisHub database client for ObsPy.

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from lxml import objectify
import pickle
import sys
import time
import urllib2


class Client(object):
    """
    SeisHub database request Client class.
    """
    def __init__(self, base_url="http://teide.geophysik.uni-muenchen.de:8080",
                 user="admin", password="admin", timeout=10):
        self.base_url = base_url
        self.waveform = _WaveformMapperClient(self)
        self.station = _StationMapperClient(self)
        self.event = _EventMapperClient(self)
        self.timeout = timeout
        # Create an OpenerDirector for Basic HTTP Authentication
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, base_url, user, password)
        auth_handler = urllib2.HTTPBasicAuthHandler(password_mgr)
        opener = urllib2.build_opener(auth_handler)
        # install globally
        urllib2.install_opener(opener)

    def ping(self):
        """
        Ping the SeisHub server.
        """
        t1 = time.time()
        urllib2.urlopen(self.base_url).read()
        return (time.time() - t1) * 1000.0

    def _fetch(self, url, *args, **kwargs):
        remoteaddr = self.base_url + url + '?'
        for key, value in kwargs.iteritems():
            if not value:
                continue
            if isinstance(value, tuple) and len(value) == 2:
                remoteaddr += 'min_' + str(key) + '=' + str(value[0]) + '&'
                remoteaddr += 'max_' + str(key) + '=' + str(value[1]) + '&'
            elif isinstance(value, list) and len(value) == 2:
                remoteaddr += 'min_' + str(key) + '=' + str(value[0]) + '&'
                remoteaddr += 'max_' + str(key) + '=' + str(value[1]) + '&'
            else:
                remoteaddr += str(key) + '=' + str(value) + '&'
        # timeout exists only for Python >= 2.6
        if sys.hexversion < 0x02060000:
            response = urllib2.urlopen(remoteaddr)
        else:
            response = urllib2.urlopen(remoteaddr, timeout=self.timeout)
        doc = response.read()

        return doc

    def _objectify(self, url, *args, **kwargs):
        doc = self._fetch(url, *args, **kwargs)
        return objectify.fromstring(doc)


class _WaveformMapperClient(object):
    """
    """
    def __init__(self, client):
        self.client = client

    def getNetworkIds(self, **kwargs):
        url = '/seismology/waveform/getNetworkIds'
        root = self.client._objectify(url, **kwargs)
        return [str(node['network']) for node in root.getchildren()]

    def getStationIds(self, **kwargs):
        url = '/seismology/waveform/getStationIds'
        root = self.client._objectify(url, **kwargs)
        return [str(node['station']) for node in root.getchildren()]

    def getLocationIds(self, **kwargs):
        url = '/seismology/waveform/getLocationIds'
        root = self.client._objectify(url, **kwargs)
        return [str(node['location']) for node in root.getchildren()]

    def getChannelIds(self, **kwargs):
        url = '/seismology/waveform/getChannelIds'
        root = self.client._objectify(url, **kwargs)
        return [str(node['channel']) for node in root.getchildren()]

    def getLatency(self, *args, **kwargs):
        """
        Gets a list of network latency values.
        
        :param network_id: Network code, e.g. 'BW'.
        :param station_id: Station code, e.g. 'MANZ'.
        :param location_id: Location code, e.g. '01'.
        :param channel_id: Channel code, e.g. 'EHE'.
        :return: List of dictionaries containing latency information.
        """
        map = ['network_id', 'station_id', 'location_id', 'channel_id']
        for i in range(len(args)):
            kwargs[map[i]] = args[i]
        url = '/seismology/waveform/getLatency'
        root = self.client._objectify(url, **kwargs)
        return [node.__dict__ for node in root.getchildren()]

    def getWaveform(self, *args, **kwargs):
        """
        Gets a Obspy Stream object.
        
        :param network_id: Network code, e.g. 'BW'.
        :param station_id: Station code, e.g. 'MANZ'.
        :param location_id: Location code, e.g. '01'.
        :param channel_id: Channel code, e.g. 'EHE'.
        :param start_datetime: start time as
            :class:`~obspy.core.utcdatetime.UTCDateTime` object.
        :param end_datetime: end time as 
            :class:`~obspy.core.utcdatetime.UTCDateTime` object
        :param apply_filter: apply filter, default False.
        :return: :class:`~obspy.core.stream.Stream` object.
        """
        map = ['network_id', 'station_id', 'location_id', 'channel_id',
               'start_datetime', 'end_datetime', 'apply_filter']
        for i in range(len(args)):
            kwargs[map[i]] = args[i]
        url = '/seismology/waveform/getWaveform'
        data = self.client._fetch(url, **kwargs)
        if data == '':
            raise Exception("No waveform data available")
        # unpickle
        stream = pickle.loads(data)
        if len(stream) == 0:
            raise Exception("No waveform data available")
        return stream

    def getPreview(self, *args, **kwargs):
        """
        Gets a preview of a Obspy Stream object.
        
        :param network_id: Network code, e.g. 'BW'.
        :param station_id: Station code, e.g. 'MANZ'.
        :param location_id: Location code, e.g. '01'.
        :param channel_id: Channel code, e.g. 'EHE'.
        :param start_datetime: start time as
            :class:`~obspy.core.utcdatetime.UTCDateTime` object.
        :param end_datetime: end time as 
            :class:`~obspy.core.utcdatetime.UTCDateTime` object
        :return: :class:`~obspy.core.stream.Stream` object.
        """
        map = ['network_id', 'station_id', 'location_id', 'channel_id',
               'start_datetime', 'end_datetime']
        for i in range(len(args)):
            kwargs[map[i]] = args[i]
        url = '/seismology/waveform/getPreview'
        data = self.client._fetch(url, **kwargs)
        if not data:
            raise Exception("No waveform data available")
        # unpickle
        stream = pickle.loads(data)
        return stream


class _BaseRESTClient(object):
    def __init__(self, client):
        self.client = client

    def getResource(self, resource_name, **kwargs):
        """
        Gets a resource.
        
        :param resource_name: Name of the resource.
        :param format: Format string, e.g. 'xml' or 'map'.
        :return: Resource
        """
        url = '/xml/' + self.package + '/' + self.resourcetype + '/' + \
              resource_name
        return self.client._fetch(url, **kwargs)

    def getXMLResource(self, resource_name, **kwargs):
        """
        Gets a XML resource.
        
        :param resource_name: Name of the resource.
        :return: Resource
        """
        url = '/xml/' + self.package + '/' + self.resourcetype + '/' + \
              resource_name
        return self.client._objectify(url, **kwargs)


class _StationMapperClient(_BaseRESTClient):
    """
    """
    package = 'seismology'
    resourcetype = 'station'

    def getList(self, *args, **kwargs):
        """
        Gets a list of station information.
        
        :param network_id: Network code, e.g. 'BW'.
        :param station_id: Station code, e.g. 'MANZ'.
        :return: List of dictionaries containing station information.
        """
        map = ['network_id', 'station_id']
        for i in range(len(args)):
            kwargs[map[i]] = args[i]
        url = '/seismology/station/getList'
        root = self.client._objectify(url, **kwargs)
        return [node.__dict__ for node in root.getchildren()]

    def getPAZ(self, network_id, station_id, datetime, location_id='',
               channel_id=''):
        """
        Get PAZ for a station at given time span.
        
        >>> c = Client()
        >>> a = c.station.getPAZ('BW', 'MANZ', '20090707', channel_id='EHZ')
        >>> a['zeros']
        [0j, 0j]
        >>> a['poles']
        [(-0.037004000000000002+0.037016j), (-0.037004000000000002-0.037016j), (-251.33000000000001+0j), (-131.03999999999999-467.29000000000002j), (-131.03999999999999+467.29000000000002j)]
        >>> a['gain']
        60077000.0
        >>> a['sensitivity']
        2516800000.0

        XXX: currently not working
        a['name']
        'Streckeisen STS-2/N seismometer'
        
        :param network_id: Network id, e.g. 'BW'.
        :param station_id: Station id, e.g. 'RJOB'.
        :param location_id: Location id, e.g. ''.
        :param channel_id: Channel id, e.g. 'EHE'.
        :param datetime: :class:`~obspy.core.utcdatetime.UTCDateTime` or
            time string.
        :return: Dictionary containing zeros, poles, gain and sensitivity.
        """
        # request station information
        station_list = self.getList(network_id=network_id,
                                    station_id=station_id, datetime=datetime)
        if not station_list:
            return {}
        # don't allow wild cards - either search over exact one node or all
        for t in ['*', '?']:
            if t in channel_id:
                channel_id = ''
            if t in location_id:
                location_id = ''

        xml_doc = station_list[0]
        # request station resource
        res = self.client.station.getXMLResource(xml_doc['resource_name'])
        base_node = res.station_control_header
        # search for nodes with correct channel and location code
        if channel_id or location_id:
            xpath_expr = "channel_identifier[channel_identifier='" + \
                channel_id + "' and location_identifier='" + location_id + "']"
            # fetch next following response_poles_and_zeros node
            xpath_expr = "channel_identifier[channel_identifier='" + \
                channel_id + "' and location_identifier='" + location_id + \
                "']/following-sibling::response_poles_and_zeros"
            paz_node = base_node.xpath(xpath_expr)[0]
            # fetch next following channel_sensitivity_node with 
            # stage_sequence_number == 0
            xpath_expr = "channel_identifier[channel_identifier='" + \
                channel_id + "' and location_identifier='" + location_id + \
                "']/following-sibling::channel_sensitivity_" + \
                "gain[stage_sequence_number='0']"
            sensitivity_node = base_node.xpath(xpath_expr)[0]
        else:
            # just take first existing nodes
            paz_node = base_node.response_poles_and_zeros[0]
            sensitivity_node = base_node.channel_sensitivity_gain[-1]
        # instrument name
        # XXX: this probably changes with a newer XSEED format
        #xpath_expr = "generic_abbreviation[abbreviation_lookup_code='" + \
        #    str(channel_node.instrument_identifier) + "']"
        #name = dict_node.xpath(xpath_expr)[0].abbreviation_description
        # poles
        poles_real = paz_node.complex_pole.real_pole[:]
        poles_imag = paz_node.complex_pole.imaginary_pole[:]
        poles = zip(poles_real, poles_imag)
        poles = [p[0] + p[1] * 1j for p in poles]
        # zeros
        zeros_real = paz_node.complex_zero.real_zero[:][:]
        zeros_imag = paz_node.complex_zero.imaginary_zero[:][:]
        zeros = zip(zeros_real, zeros_imag)
        zeros = [p[0] + p[1] * 1j for p in zeros]
        # gain
        gain = paz_node.A0_normalization_factor
        # sensitivity
        sensitivity = sensitivity_node.sensitivity_gain
        return {'poles': poles, 'zeros': zeros, 'gain': gain,
                'sensitivity': sensitivity}# 'name': name}


class _EventMapperClient(_BaseRESTClient):
    """
    """
    package = 'seismology'
    resourcetype = 'event'

    def getList(self, *args, **kwargs):
        """
        Gets a list of event information.
        
        :return: List of dictionaries containing event information.
        """
        url = '/seismology/event/getList'
        root = self.client._objectify(url, **kwargs)
        return [node.__dict__ for node in root.getchildren()]


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
