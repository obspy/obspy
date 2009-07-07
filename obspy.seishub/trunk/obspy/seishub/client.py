# -*- coding: utf-8 -*-

from lxml import etree, objectify
import sys
import tempfile
import obspy
import urllib


class Client(object):
    """
    """
    def __init__(self, base_url, *args, **kwargs):
        self.base_url = base_url
        self.waveform = _WaveformMapperClient(self)
        self.station = _StationMapperClient(self)

    def _fetch(self, url, *args, **kwargs):
        remoteaddr = self.base_url + url + '?'
        for key, value in kwargs.iteritems():
            if not value:
                continue
            remoteaddr += str(key) + '=' + str(value) + '&'
        doc = urllib.urlopen(remoteaddr).read()
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
        return [str(node.getchildren()[0]) for node in root.getchildren()]

    def getStationIds(self, **kwargs):
        url = '/seismology/waveform/getStationIds'
        root = self.client._objectify(url, **kwargs)
        return [str(node.getchildren()[0]) for node in root.getchildren()]

    def getLocationIds(self, **kwargs):
        url = '/seismology/waveform/getLocationIds'
        root = self.client._objectify(url, **kwargs)
        return [str(node.getchildren()[0]) for node in root.getchildren()]

    def getChannelIds(self, **kwargs):
        url = '/seismology/waveform/getChannelIds'
        root = self.client._objectify(url, **kwargs)
        return [str(node.getchildren()[0]) for node in root.getchildren()]

    def getLatency(self, **kwargs):
        """
        Gets a list of network latency values.
        
        @param network_id: Network code, e.g. 'BW'.
        @param station_id: Station code, e.g. 'MANZ'.
        @param location_id: Location code, e.g. '01'.
        @param channel_id: Channel code, e.g. 'EHE'.
        @return: List of dictionaries containing latency information.
        """
        url = '/seismology/waveform/getLatency'
        root = self.client._objectify(url, **kwargs)
        return [node.__dict__ for node in root.getchildren()]

    def getWaveform(self, *args, **kwargs):
        """
        Gets a list of network latency values.
        
        @param network_id: Network code, e.g. 'BW'.
        @param station_id: Station code, e.g. 'MANZ'.
        @param location_id: Location code, e.g. '01'.
        @param channel_id: Channel code, e.g. 'EHE'.
        @return: List of dictionaries containing latency information.
        """
        map = ['network_id', 'station_id', 'location_id', 'channel_id',
               'start_datetime', 'end_datetime']
        for i in range(len(args)):
            kwargs[map[i]] = args[i]
        url = '/seismology/waveform/getWaveform'
        data = self.client._fetch(url, **kwargs)
        if not data:
            return None
        tf = tempfile.NamedTemporaryFile(mode='wb')
        tf.write(data)
        tf.seek(0)
        trace = obspy.read(tf.name)
        tf.close()
        return trace



class _StationMapperClient(object):
    """
    """
    def __init__(self, client):
        self.client = client

    def getList(self, **kwargs):
        """
        Gets a list of station information.
        
        @param network_id: Network code, e.g. 'BW'.
        @param station_id: Station code, e.g. 'MANZ'.
        @return: List of dictionaries containing station information.
        """
        url = '/seismology/station/getStationList'
        root = self.client._objectify(url, **kwargs)
        return [node.__dict__ for node in root.getchildren()]

    def getResource(self, network_id, station_id, **kwargs):
        """
        Gets a station resource.
        
        @param network_id: Network code, e.g. 'BW'.
        @param station_id: Station code, e.g. 'MANZ'.
        @param format: Format string, e.g. 'xml' or 'map'.
        @return: Resource
        """
        items = self.getList(limit=1, network_id=network_id,
                             station_id=station_id, **kwargs)
        resource_name = items[0]['resource_name']
        url = '/xml/seismology/station/' + resource_name
        return self.client._fetch(url, **kwargs)

    def getXMLResource(self, network_id, station_id, **kwargs):
        """
        Gets a station XML resource.
        
        @param network_id: Network code, e.g. 'BW'.
        @param station_id: Station code, e.g. 'MANZ'.
        @return: Resource
        """
        items = self.getList(limit=1, network_id=network_id,
                             station_id=station_id, **kwargs)
        resource_name = items[0]['resource_name']
        url = '/xml/seismology/station/' + resource_name
        return self.client._objectify(url, **kwargs)
