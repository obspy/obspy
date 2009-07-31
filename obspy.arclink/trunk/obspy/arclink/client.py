# -*- coding: utf-8 -*-

from lxml import objectify
from obspy.core import read, Stream, UTCDateTime
from obspy.core.util import NamedTemporaryFile, AttribDict
from telnetlib import Telnet
import time, sys


class Client(Telnet):
    """
    """
    status_timeout = 1
    status_delay = 0.1

    def __init__(self, host="webdc.eu", port=18001, timeout=None,
                 user="Anonymous", institution="Anonymous", debug=False):
        self.user = user
        self.institution = institution
        if sys.hexversion < 0x020600F0:
            Telnet.__init__(self, host, port)
        else:
            Telnet.__init__(self, host, port, timeout)
        # silent connection check
        self.debug = False
        self._hello()
        self.debug = debug

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
            return ''
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

    def _objectify(self, *args, **kwargs):
        doc = self._fetch(*args, **kwargs)
        return objectify.fromstring(doc)

    def getWaveform(self, network_id, station_id, location_id, channel_id,
                    start_datetime, end_datetime):
        """
        Gets a L{obspy.Stream} object.
        
        @param network_id: Network code, e.g. 'BW'.
        @param station_id: Station code, e.g. 'MANZ'.
        @param location_id: Location code, e.g. '01'.
        @param channel_id: Channel code, e.g. 'EHE'.
        @param start_datetime: start time as L{obspy.UTCDateTime} object.
        @param end_datetime: end time as L{obspy.UTCDateTime} object.
        @return: L{obspy.Stream} object.
        """
        rtype = 'REQUEST WAVEFORM format=MSEED'
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
        xml_doc = self._objectify(rtype, [rdata])
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
