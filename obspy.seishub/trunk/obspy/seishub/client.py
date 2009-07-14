# -*- coding: utf-8 -*-

from lxml import etree, objectify
import obspy
import sys
import os
import tempfile
import urllib


class Client(object):
    """
    """
    def __init__(self, base_url, *args, **kwargs):
        self.base_url = base_url
        self.waveform = _WaveformMapperClient(self)
        self.station = _StationMapperClient(self)
        self.event = _EventMapperClient(self)

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

    def getLatency(self, *args, **kwargs):
        """
        Gets a list of network latency values.
        
        @param network_id: Network code, e.g. 'BW'.
        @param station_id: Station code, e.g. 'MANZ'.
        @param location_id: Location code, e.g. '01'.
        @param channel_id: Channel code, e.g. 'EHE'.
        @return: List of dictionaries containing latency information.
        """
        map = ['network_id', 'station_id', 'location_id', 'channel_id']
        for i in range(len(args)):
            kwargs[map[i]] = args[i]
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

        class TempFile(object):
            def __init__(self, fd, fname):
                self._fileobj = os.fdopen(fd, 'w+b')
                self.name = fname
            def __getattr__(self, attr):
                return getattr(self._fileobj, attr)

        def mktempfile(dir=None, suffix='.tmp'):
            return TempFile(*tempfile.mkstemp(dir=dir, suffix=suffix))


        tf = mktempfile()
        try:
            tf.write(data)
            tf.seek(0)
            trace = obspy.read(tf.name, 'MSEED')
        finally:
            tf.close()
        return trace


class _BaseRESTClient(object):
    def __init__(self, client):
        self.client = client

    def getResource(self, resource_name, **kwargs):
        """
        Gets a resource.
        
        @param resource_name: Name of the resource.
        @param format: Format string, e.g. 'xml' or 'map'.
        @return: Resource
        """
        url = '/xml/' + self.package + '/' + self.resourcetype + '/' + \
              resource_name
        return self.client._fetch(url, **kwargs)

    def getXMLResource(self, resource_name, **kwargs):
        """
        Gets a XML resource.
        
        @param resource_name: Name of the resource.
        @return: Resource
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
        
        @param network_id: Network code, e.g. 'BW'.
        @param station_id: Station code, e.g. 'MANZ'.
        @return: List of dictionaries containing station information.
        """
        map = ['network_id', 'station_id']
        for i in range(len(args)):
            kwargs[map[i]] = args[i]
        url = '/seismology/station/getList'
        root = self.client._objectify(url, **kwargs)
        return [node.__dict__ for node in root.getchildren()]


class _EventMapperClient(_BaseRESTClient):
    """
    """
    package = 'seismology'
    resourcetype = 'event'

    def getList(self, *args, **kwargs):
        """
        Gets a list of event information.
        
        @return: List of dictionaries containing event information.
        """
        url = '/seismology/event/getList'
        root = self.client._objectify(url, **kwargs)
        return [node.__dict__ for node in root.getchildren()]
