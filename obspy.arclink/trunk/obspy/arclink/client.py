# -*- coding: utf-8 -*-

from lxml import objectify, etree
from obspy.core import read, Stream, UTCDateTime
from obspy.core.util import NamedTemporaryFile, AttribDict
from telnetlib import Telnet
import os
import sys
import time


class ArcLinkException(Exception):
    pass


class Client(Telnet):
    """
    """
    status_timeout = 1
    status_delay = 0.1

    def __init__(self, host="webdc.eu", port=18001,
                 timeout=20, user="Anonymous", institution="Anonymous",
                 debug=False):
        self.user = user
        self.institution = institution
        # timeout exists only for Python >= 2.6
        if sys.hexversion < 0x02060000:
            Telnet.__init__(self, host, port)
        else:
            Telnet.__init__(self, host, port, timeout)
        # silent connection check
        self.debug = False
        self._hello()
        self.debug = debug
        # fetch and parse inventory schema only once
        path = os.path.dirname(__file__)
        file = open(os.path.join(path, 'xsd', 'inventory.xsd'))
        schema = etree.XMLSchema(file=file)
        self.inventory_parser = objectify.makeparser(schema=schema)
        file.close()

    def writeln(self, buffer):
        self.write(buffer + '\n')

    def write(self, buffer):
        if self.debug:
            print '>>> ' + buffer.strip()
        Telnet.write(self, buffer)

    def readln(self, value=''):
        line = self.read_until(value + '\r\n', self.status_timeout)
        line = line.strip()
        if self.debug:
            print '... ' + line
        return line

    def _hello(self):
        if sys.hexversion < 0x020600F0:
            self.open(self.host, self.port)
        else:
            self.open(self.host, self.port, self.timeout)
        self.writeln('HELLO')
        self.version = self.readln(')')
        self.node = self.readln()
        self.writeln('USER %s' % self.user)
        self.readln('OK')
        self.writeln('INSTITUTION %s' % self.institution)
        self.readln('OK')

    def _bye(self):
        self.writeln('BYE')
        self.close()

    def _fetch(self, request_type, request_data):
        self._hello()
        self.writeln(request_type)
        for line in request_data:
            self.writeln(line)
        self.writeln('END')
        self.readln('OK')
        self.writeln('STATUS')
        req_id = int(self.readln())
        while 1:
            self.writeln('STATUS %d' % req_id)
            xml_doc = self.readln()
            if 'ready="true"' in xml_doc:
                self.read_until('\r\n')
                break
            time.sleep(self.status_delay)
        if 'id="NODATA" status="NODATA"' in xml_doc:
            self.writeln('PURGE %d' % req_id)
            self._bye()
            raise ArcLinkException('No data available')
        if '<line content' not in xml_doc:
            self.writeln('PURGE %d' % req_id)
            self._bye()
            raise ArcLinkException('No content')
        self.writeln('DOWNLOAD %d' % req_id)
        length = int(self.readln())
        data = ''
        i = 0
        while 1:
            data += self.rawq_getchar()
            i += 1
            if i > length and 'END' in data:
                break
        data = data[:-3]
        if len(data) != length:
            raise Exception('Wrong length!')
        if self.debug:
            print data
        self.writeln('PURGE %d' % req_id)
        self._bye()
        return data

    def saveWaveform(self, filename, network_id, station_id, location_id,
                     channel_id, start_datetime, end_datetime, format="MSEED"):
        """
        Writes a fetched waveform into a file.
        
        @param filename: String containing the filename.
        @param network_id: Network code, e.g. 'BW'.
        @param station_id: Station code, e.g. 'MANZ'.
        @param location_id: Location code, e.g. '01'.
        @param channel_id: Channel code, e.g. 'EHE'.
        @param start_datetime: start time as L{obspy.UTCDateTime} object.
        @param end_datetime: end time as L{obspy.UTCDateTime} object.
        @param format: 'FSEED', 'MSEED', or 'XSEED'.
        @return: L{obspy.Stream} object.
        """
        rtype = 'REQUEST WAVEFORM format=%s' % format
        # adding one second to start and end time to ensure right date times
        rdata = "%s %s %s %s %s %s" % ((start_datetime - 1).formatArcLink(),
                                       (end_datetime + 1).formatArcLink(),
                                       network_id, station_id, channel_id,
                                       location_id)
        data = self._fetch(rtype, [rdata])
        fh = open(filename, "wb")
        fh.write(data)
        fh.close()

    def getWaveform(self, network_id, station_id, location_id, channel_id,
                    start_datetime, end_datetime, format="MSEED"):
        """
        Gets a L{obspy.Stream} object.
        
        @param network_id: Network code, e.g. 'BW'.
        @param station_id: Station code, e.g. 'MANZ'.
        @param location_id: Location code, e.g. '01'.
        @param channel_id: Channel code, e.g. 'EHE'.
        @param start_datetime: start time as L{obspy.UTCDateTime} object.
        @param end_datetime: end time as L{obspy.UTCDateTime} object.
        @param format: 'FSEED' or 'MSEED' ('XSEED' is documented, but not yet 
            implemented in ArcLink).
        @return: L{obspy.Stream} object.
        """
        rtype = 'REQUEST WAVEFORM format=%s' % format
        # adding one second to start and end time to ensure right date times
        rdata = "%s %s %s %s %s %s" % ((start_datetime - 1).formatArcLink(),
                                       (end_datetime + 1).formatArcLink(),
                                       network_id, station_id, channel_id,
                                       location_id)
        data = self._fetch(rtype, [rdata])
        if data:
            tf = NamedTemporaryFile()
            tf.write(data)
            tf.seek(0)
            try:
                stream = read(tf.name, 'MSEED')
            finally:
                tf.close()
        else:
            stream = Stream()
        # trim stream
        stream.trim(start_datetime, end_datetime)
        return stream

    def saveResponse(self, filename, network_id, station_id, location_id,
                     channel_id, start_datetime, end_datetime, format='SEED'):
        """
        Writes a response information into a file.
        
        @param network_id: Network code, e.g. 'BW'.
        @param station_id: Station code, e.g. 'MANZ'.
        @param location_id: Location code, e.g. '01'.
        @param channel_id: Channel code, e.g. 'EHE'.
        @param start_datetime: start time as L{obspy.UTCDateTime} object.
        @param end_datetime: end time as L{obspy.UTCDateTime} object.
        @param format: 'SEED' ('XSEED' is documented, but not yet implemented 
            in ArcLink).
        """
        rtype = 'REQUEST RESPONSE format=%s' % format
        # adding one second to start and end time to ensure right date times
        rdata = "%s %s %s %s %s %s" % ((start_datetime - 1).formatArcLink(),
                                       (end_datetime + 1).formatArcLink(),
                                       network_id, station_id, channel_id,
                                       location_id)
        data = self._fetch(rtype, [rdata])
        fh = open(filename, "wb")
        fh.write(data)
        fh.close()

    def getNetworks(self, start_datetime, end_datetime):
        """
        Returns a dictionary of available networks within the given time span.
        
        @param start_datetime: start time as L{obspy.UTCDateTime} object.
        @param end_datetime: end time as L{obspy.UTCDateTime} object.
        @return: dictionary of network data.
        """
        rtype = 'REQUEST INVENTORY'
        rdata = "%s %s *" % (start_datetime.formatArcLink(),
                             end_datetime.formatArcLink())
        # fetch plain xml document
        xml_doc = self._fetch(rtype, [rdata])
        # generate object by using XML schema
        xml_doc = objectify.fromstring(xml_doc, self.inventory_parser)

        data = AttribDict()
        if not xml_doc.countchildren():
            return data
        for network in xml_doc.network:
            # XXX: not secure - map it manually
            temp = AttribDict(dict(network.attrib))
            temp['remark'] = str(network.remark)
            try:
                temp.start = UTCDateTime(temp.start)
            except:
                temp.start = None
            try:
                temp.end = UTCDateTime(temp.end)
            except:
                temp.end = None
            data[network.attrib['code']] = temp
        return data
