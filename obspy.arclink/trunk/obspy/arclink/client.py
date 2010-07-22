# -*- coding: utf-8 -*-
"""
ArcLink client.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from lxml import objectify, etree
from obspy.core import read, Stream, UTCDateTime
from obspy.core.util import NamedTemporaryFile, AttribDict, complexifyString
from telnetlib import Telnet
import os
import sys
import time


class ArcLinkException(Exception):
    pass


class Client(Telnet):
    """
    The ArcLink/WebDC client.

    Parameters
    ----------
    host : string, optional
        Host name of the remote ArcLink server (default host is 'webdc.eu').
    port : int, optional
        Port of the remote ArcLink server (default port is 18001).
    timeout : int, optional
        Seconds before a connection timeout is raised (default is 20 seconds).
        This works only for Python >= 2.6.x.
    user : string, optional
        The user name used for authentication with the ArcLink server (default
        is an 'Anonymous' for accessing public ArcLink server).
    password : string, optional
        A password used for authentication with the ArcLink server (default is
        an empty string).
    institution : string, optional
        A string containing the name of the institution of the requesting
        person (default is an 'Anonymous').
    debug : boolean, optional
        Enables verbose output of the connection handling (default is False). 

    Notes
    -----
    The following ArcLink servers may be accessed via ObsPy:

    Public servers:
      * WebDC servers: webdc.eu:18001, webdc:18002

    Further mirrors listed at webdc.eu (restricted access only):
      * ODC Server:  bhlsa03.knmi.nl:18001
      * INGV Server: discovery1.rm.ingv.it:18001
      * IPGP Server: geosrt2.ipgp.fr:18001
    """
    status_timeout = 1
    status_delay = 0.1

    def __init__(self, host="webdc.eu", port=18001, timeout=20,
                 user="Anonymous", password="", institution="Anonymous",
                 debug=False):
        """
        """
        self.user = user
        self.password = password
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

    def _writeln(self, buffer):
        Telnet.write(self, buffer + '\n')
        if self.debug:
            print '>>> ' + buffer

    def _readln(self, value=''):
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
        self._writeln('HELLO')
        self.version = self._readln(')')
        self.node = self._readln()
        if self.password:
            self._writeln('USER %s %s' % (self.user, self.password))
        else:
            self._writeln('USER %s' % self.user)
        self._readln('OK')
        self._writeln('INSTITUTION %s' % self.institution)
        self._readln('OK')

    def _bye(self):
        self._writeln('BYE')
        self.close()

    def _fetch(self, request_type, request_data):
        self._hello()
        self._writeln(request_type)
        for line in request_data:
            self._writeln(line)
        self._writeln('END')
        self._readln('OK')
        self._writeln('STATUS')
        req_id = int(self._readln())
        while 1:
            self._writeln('STATUS %d' % req_id)
            xml_doc = self._readln()
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
            self._writeln('PURGE %d' % req_id)
            self._bye()
            # parse XML for error message
            xml_doc = objectify.fromstring(xml_doc[:-3])
            raise ArcLinkException(xml_doc.request.volume.line.get('message'))
        # XXX: safeguard as long not all status messages are covered 
        if '<line content' not in xml_doc:
            self._writeln('PURGE %d' % req_id)
            self._bye()
            raise ArcLinkException('No content')
        self._writeln('DOWNLOAD %d' % req_id)
        fd = self.get_socket().makefile('rb+')
        length = int(fd.readline(100).strip())
        data = ''
        while len(data) < length:
            buf = fd.read(min(4096, length - len(data)))
            data += buf
        buf = fd.readline(100).strip()
        if buf != "END" or len(data) != length:
            raise Exception('Wrong length!')
        if self.debug:
            print "%d bytes of data read" % len(data)
        self._writeln('PURGE %d' % req_id)
        self._bye()
        return data

    def saveWaveform(self, filename, network_id, station_id, location_id,
                     channel_id, start_datetime, end_datetime, format="MSEED",
                     compressed=True):
        """
        Writes a retrieved waveform directly into a file.

        Parameters
        ----------
        filename : string
            Name of the output file.
        network_id : string
            Network code, e.g. 'BW'.
        station_id : string
            Station code, e.g. 'MANZ'.
        location_id : string
            Location code, e.g. '01'.
        channel_id : string
            Channel code, e.g. 'EHE'.
        start_datetime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        end_datetime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        format : ['FSEED' | 'MSEED'], optional
            Output format. Either as full SEED ('FSEED') or Mini-SEED ('MSEED')
            volume (default is an 'MSEED'). 
            .. note:: 
                Format 'XSEED' is documented, but not yet implemented in
                ArcLink.
        compressed : boolean, optional 
            Request compressed files from ArcLink server (default is True).
        """
        rtype = 'REQUEST WAVEFORM format=%s' % format
        if compressed:
            try:
                import bz2
            except:
                compressed = False
            else:
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
        Retrieve waveform via ArcLink and returns a ObsPy Stream object.

        Parameters
        ----------
        filename : string
            Name of the output file.
        network_id : string
            Network code, e.g. 'BW'.
        station_id : string
            Station code, e.g. 'MANZ'.
        location_id : string
            Location code, e.g. '01'.
        channel_id : string
            Channel code, e.g. 'EHE'.
        start_datetime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        end_datetime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        format : ['FSEED' | 'MSEED'], optional
            Output format. Either as full SEED ('FSEED') or Mini-SEED ('MSEED')
            volume (default is an 'MSEED'). 
            .. note:: 
                Format 'XSEED' is documented, but not yet implemented in
                ArcLink.
        compressed : boolean, optional 
            Request compressed files from ArcLink server (default is True).

        Returns
        -------
        :class:`~obspy.core.stream.Stream`
            Returns an ObsPy Stream object.
        """
        rtype = 'REQUEST WAVEFORM format=%s' % format
        if compressed:
            try:
                import bz2
            except:
                compressed = False
            else:
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
            # we need to create a temporary file, as libmseed only accepts
            # filenames and not Python file pointers 
            tf = NamedTemporaryFile()
            tf.write(data)
            tf.seek(0)
            try:
                stream = read(tf.name, 'MSEED')
            finally:
                tf.close()
            # remove temporary file:
            try:
                os.remove(tf.name)
            except:
                pass
        else:
            stream = Stream()
        # trim stream
        stream.trim(start_datetime, end_datetime)
        return stream

    def getPAZ(self, network_id, station_id, location_id,
                    channel_id, start_datetime, end_datetime):
        """
        Returns poles, zeros, gain and sensitivity of a single channel.

        :param network_id: Network code, e.g. 'BW'.
        :param station_id: Station code, e.g. 'MANZ'.
        :param location_id: Location code, e.g. '01'.
        :param channel_id: Channel code, e.g. 'EHE'.
        :param start_datetime: start time as L{obspy.UTCDateTime} object.
        :param end_datetime: end time as L{obspy.UTCDateTime} object.
        :return: dictionary containing PAZ information
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
        if not xml_doc.countchildren():
            return {}
        pazs = {}
        for resp_paz in xml_doc.resp_paz:
            paz = {}
            # instrument name
            instrument = resp_paz.attrib['name']
            # parsing gain
            paz['gain'] = float(resp_paz.attrib['norm_fac'])
            # parsing zeros
            paz['zeros'] = []
            temp = str(resp_paz.zeros).strip().replace(' ', '')
            temp = temp.replace(')(', ') (')
            for zeros in temp.split():
                paz['zeros'].append(complexifyString(zeros))
            if len(paz['zeros']) != int(resp_paz.attrib['nzeros']):
                raise ArcLinkException('Could not parse all zeros')
            # parsing poles
            paz['poles'] = []
            temp = str(resp_paz.poles).strip().replace(' ', '')
            temp = temp.replace(')(', ') (')
            for poles in temp.split():
                paz['poles'].append(complexifyString(poles))
            if len(paz['poles']) != int(resp_paz.attrib['npoles']):
                raise ArcLinkException('Could not parse all poles')
            # parsing sensitivity
            component = xml_doc.network.station.seis_stream.component
            if len(component) > 1:
                msg = 'Currently supporting only a single channel, e.g. EHZ'
                raise ArcLinkException(msg)
            try:
                paz['sensitivity'] = float(component[0].attrib['gain'])
            except:
                paz['sensitivity'] = float(resp_paz.attrib['gain'])
            pazs[instrument] = paz
        return pazs

    def saveResponse(self, filename, network_id, station_id, location_id,
                     channel_id, start_datetime, end_datetime, format='SEED'):
        """
        Writes a response information into a file.

        :param network_id: Network code, e.g. 'BW'.
        :param station_id: Station code, e.g. 'MANZ'.
        :param location_id: Location code, e.g. '01'.
        :param channel_id: Channel code, e.g. 'EHE'.
        :param start_datetime: start time as L{obspy.UTCDateTime} object.
        :param end_datetime: end time as L{obspy.UTCDateTime} object.
        :param format: 'SEED' ('XSEED' is documented, but not yet implemented 
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

        Currently the time span is ignored by the ArcLink servers, therefore
        all possible networks are returned.

        :param start_datetime: start time as L{obspy.UTCDateTime} object.
        :param end_datetime: end time as L{obspy.UTCDateTime} object.
        :return: dictionary of network data.
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
