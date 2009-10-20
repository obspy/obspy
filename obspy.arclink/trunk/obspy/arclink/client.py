# -*- coding: utf-8 -*-

from lxml import objectify, etree
from obspy.core import read, Stream, UTCDateTime
from obspy.core.util import NamedTemporaryFile, AttribDict, complexifyString
from telnetlib import Telnet
import bz2
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
    # possible mirrors listed in webdc.eu, content of all is restricted
    # ODC Server  bhlsa03.knmi.nl:18001 
    # INGV Server discovery1.rm.ingv.it:18001 
    # IPGP Server geosrt2.ipgp.fr:18001 
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
        # check for errors
        # XXX: not everything implemented yet
        #     = OK - request sucessfully processed, data available
        #     = NODATA - no processing errors, but data not available
        #     = WARN - processing errors, some downloadable data available
        #     = ERROR - processing errors, no downloadable data available
        #     = RETRY - temporarily no data available
        #     = DENIED - access to data denied for the user
        #     = CANCEL - processing cancelled (eg., by operator)
        #     = MESSAGE <any_string> - error message in case of WARN or
        #           ERROR, but can be used regardless of status (the last 
        #           message is shown in STATUS response)
        #     = SIZE <n> - data size. In case of volume, it must be the 
        #           exact size of downloadable product.
        if 'id="NODATA"' in xml_doc or 'id="ERROR"' in xml_doc:
            # error or no data
            self.writeln('PURGE %d' % req_id)
            self._bye()
            # parse XML for error message
            xml_doc = objectify.fromstring(xml_doc[:-3])
            raise ArcLinkException(xml_doc.request.volume.line.get('message'))
        # XXX: safeguard as long not all status messages are covered 
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
                     channel_id, start_datetime, end_datetime, format="MSEED",
                     compressed=True):
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
        @param compressed: Request compressed files from ArcLink server.
        @return: L{obspy.Stream} object.
        """
        rtype = 'REQUEST WAVEFORM format=%s' % format
        if compressed:
            rtype += " compression=bzip2"
        # adding one second to start and end time to ensure right date times
        rdata = "%s %s %s %s %s %s" % ((start_datetime - 1).formatArcLink(),
                                       (end_datetime + 1).formatArcLink(),
                                       network_id, station_id, channel_id,
                                       location_id)
        data = self._fetch(rtype, [rdata])
        if compressed:
            data = bz2.decompress(data)
        fh = open(filename, "wb")
        fh.write(data)
        fh.close()

    def getWaveform(self, network_id, station_id, location_id, channel_id,
                    start_datetime, end_datetime, format="MSEED",
                    compressed=True):
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
        @param compressed: Request compressed files from ArcLink server.
        @return: L{obspy.Stream} object.
        """
        rtype = 'REQUEST WAVEFORM format=%s' % format
        if compressed:
            rtype += " compression=bzip2"
        # adding one second to start and end time to ensure right date times
        rdata = "%s %s %s %s %s %s" % ((start_datetime - 1).formatArcLink(),
                                       (end_datetime + 1).formatArcLink(),
                                       network_id, station_id, channel_id,
                                       location_id)
        data = self._fetch(rtype, [rdata])
        if data:
            if compressed:
                data = bz2.decompress(data)
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

    def getPAZ(self, network_id, station_id, location_id,
                    channel_id, start_datetime, end_datetime):
        """
        Returns poles, zeros, gain and sensitivity as dictionary.
        
        @param network_id: Network code, e.g. 'BW'.
        @param station_id: Station code, e.g. 'MANZ'.
        @param location_id: Location code, e.g. '01'.
        @param channel_id: Channel code, e.g. 'EHE'.
        @param start_datetime: start time as L{obspy.UTCDateTime} object.
        @param end_datetime: end time as L{obspy.UTCDateTime} object.
        @return: dictionary containing PAZ information
        """
        rtype = 'REQUEST INVENTORY instruments=true'
        rdata = "%s %s %s %s %s %s" % (start_datetime.formatArcLink(),
                             end_datetime.formatArcLink(),
                             network_id,
                             location_id,
                             station_id,
                             channel_id)
        # fetch plain xml document
        xml_doc = self._fetch(rtype, [rdata])
        # generate object by using XML schema
        xml_doc = objectify.fromstring(xml_doc, self.inventory_parser)
        paz = {}
        if not xml_doc.countchildren():
            return paz
        if len(xml_doc.resp_paz) > 1:
            raise ArcLinkException('Currently support only parsing of'
                'single network and station BW MANZ')
        resp_paz = xml_doc.resp_paz[0]
        # parsing gain
        paz['gain'] = float(resp_paz.attrib['norm_fac'])
        # parsing zeros
        paz['zeros'] = []
        for zeros in str(resp_paz.zeros).strip().split():
            paz['zeros'].append(complexifyString(zeros))
        if len(paz['zeros']) != int(resp_paz.attrib['nzeros']):
            raise ArcLinkException('Could not parse all zeros')
        # parsing poles
        paz['poles'] = []
        for poles in str(resp_paz.poles).strip().split():
            paz['poles'].append(complexifyString(poles))
        if len(paz['poles']) != int(resp_paz.attrib['npoles']):
            raise ArcLinkException('Could not parse all poles')
        # parsing sensitivity
        component = xml_doc.network.station.seis_stream.component
        if len(component) > 1:
            raise ArcLinkException('Currently support only parsing of'
                'single channel, e.g. EHZ')
        paz['sensitivity'] = float(component[0].attrib['gain'])
        return paz

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
