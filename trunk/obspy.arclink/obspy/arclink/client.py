# -*- coding: utf-8 -*-
"""
ArcLink/WebDC client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from copy import deepcopy
from lxml import objectify, etree
from obspy.core import read, Stream, UTCDateTime
from obspy.core.util import NamedTemporaryFile, AttribDict, complexifyString
from telnetlib import Telnet
import os
import sys
import time


ROUTING_NS_1_0 = "http://geofon.gfz-potsdam.de/ns/Routing/1.0/"
ROUTING_NS_0_1 = "http://geofon.gfz-potsdam.de/ns/routing/0.1/"
INVENTORY_NS_1_0 = "http://geofon.gfz-potsdam.de/ns/Inventory/1.0/"
INVENTORY_NS_0_2 = "http://geofon.gfz-potsdam.de/ns/inventory/0.2/"


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
    command_delay : float, optional
        Delay between each command send to the ArcLink server (default is 0).
    plain_status_allowed : boolean, optional
        Certain ArcLink versions do not allow a plain STATUS request. Set this
        to False if you experience a endless loop during a request. Default is
        True.

    Notes
    -----
    The following ArcLink servers may be accessed via ObsPy:

    Public servers:
      * WebDC servers: webdc.eu:18001, webdc:18002

    Further mirrors listed at webdc.eu (partly restricted access only):
      * ODC Server:  bhlsa03.knmi.nl:18001
      * INGV Server: eida.rm.ingv.it:18001
      * IPGP Server: geosrt2.ipgp.fr:18001
    """
    status_timeout = 2
    status_delay = 0.1

    def __init__(self, host="webdc.eu", port=18002, timeout=20,
                 user="ObsPy client", password="", institution="Anonymous",
                 debug=False, command_delay=0, plain_status_allowed=True):
        """
        """
        self.user = user
        self.password = password
        self.institution = institution
        self.command_delay = command_delay
        self.init_host = host
        self.init_port = port
        self.plain_status_allowed = plain_status_allowed
        # timeout exists only for Python >= 2.6
        if sys.hexversion < 0x02060000:
            Telnet.__init__(self, host, port)
        else:
            Telnet.__init__(self, host, port, timeout)
        # silent connection check
        self.debug = False
        self._hello()
        self.debug = debug
        if self.debug:
            print('\nConnected to %s:%d' % (self.host, self.port))

    def _writeln(self, buffer):
        if self.command_delay:
            time.sleep(self.command_delay)
        Telnet.write(self, buffer + '\n')
        if self.debug:
            print('>>> ' + buffer)

    def _readln(self, value=''):
        line = self.read_until(value + '\r\n', self.status_timeout)
        line = line.strip()
        if value not in line:
            print "TIMEOUT!!! %s" % value
        if self.debug:
            print('... ' + line)
        return line

    def _hello(self):
        if sys.hexversion < 0x020600F0:
            self.open(self.host, self.port)
        else:
            self.open(self.host, self.port, self.timeout)
        self._writeln('HELLO')
        self.version = self._readln(')')
        # certain ArcLink versions do not allow a plain STATUS request
        if 'ArcLink v1.2 (2010.256)' in self.version:
            self.plain_status_allowed = False
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

    def _fetch(self, request_type, request_data, route=True):
        # skip routing on request
        if not route:
            return self._request(request_type, request_data)
        # using route
        routes = self.getRouting(network=request_data[2],
                                 station=request_data[3],
                                 starttime=request_data[0],
                                 endtime=request_data[1])
        # check if route for network and station combination exists
        id = request_data[2] + '.' + request_data[3]
        if id in routes.keys() and routes[id] == []:
            # we are at the responsible ArcLink node 
            return self._request(request_type, request_data)
        # check if route for network exists
        id = request_data[2] + '.'
        if id not in routes.keys():
            # retry first ArcLink node if host and port have been changed
            if self.host != self.init_host and self.port != self.init_port:
                self.host = self.init_host
                self.port = self.init_port
                if self.debug:
                    print('\nRequesting %s:%d' % (self.host, self.port))
                return self._fetch(request_type, request_data, route)
            msg = 'Could not find route to %s.%s'
            raise ArcLinkException(msg % (request_data[2], request_data[3]))
        # route for network id exists
        routes = routes[id]
        routes.sort(lambda x, y: cmp(x['priority'], y['priority']))
        for route in routes:
            self.host = route['host']
            self.port = route['port']
            if self.debug:
                print('\nRequesting %s:%d' % (self.host, self.port))
            # only use timeout from python2.6
            if sys.hexversion < 0x020600F0:
                self.open(self.host, self.port)
            else:
                self.open(self.host, self.port, self.timeout)
            try:
                return self._request(request_type, request_data)
            except ArcLinkException:
                raise
            except Exception:
                raise
        msg = 'Could not find route to %s.%s'
        raise ArcLinkException(msg % (request_data[2], request_data[3]))

    def _request(self, request_type, request_data):
        self._hello()
        self._writeln(request_type)
        if not self.plain_status_allowed:
            self._readln('OK')
        # create request string
        # adding one second to start and end time to ensure right date times
        out = (request_data[0] - 1).formatArcLink() + ' '
        out += (request_data[1] + 1).formatArcLink() + ' '
        out += ' '.join([str(i) for i in request_data[2:]])
        self._writeln(out)
        self._writeln('END')
        if self.plain_status_allowed:
            self._readln('OK')
            self._writeln('STATUS')
        while 1:
            status = self._readln()
            try:
                req_id = int(status)
            except:
                if 'ERROR' in status:
                    self._bye()
                    raise ArcLinkException('Error in request')
                pass
            else:
                break
        while 1:
            self._writeln('STATUS %d' % req_id)
            xml_doc = self._readln()
            if 'ready="true"' in xml_doc:
                self.read_until('\r\n', self.status_timeout)
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
        for err_code in ['DENIED', 'CANCELLED', 'CANCEL', 'ERROR']:
            err_str = 'status="%s"' % (err_code)
            if err_str in xml_doc:
                # cleanup
                self._writeln('PURGE %d' % req_id)
                self._bye()
                # parse XML for reason
                xml_doc = objectify.fromstring(xml_doc[:-3])
                msg = xml_doc.request.volume.line.get('message')
                raise ArcLinkException("%s %s" % (err_code, msg))
        if 'status="NODATA"' in xml_doc:
            # cleanup
            self._writeln('PURGE %d' % req_id)
            self._bye()
            raise ArcLinkException('No data available')
        elif 'id="NODATA"' in xml_doc or 'id="ERROR"' in xml_doc:
            # cleanup
            self._writeln('PURGE %d' % req_id)
            self._bye()
            # parse XML for error message
            xml_doc = objectify.fromstring(xml_doc[:-3])
            raise ArcLinkException(xml_doc.request.volume.line.get('message'))
        elif '<line content' not in xml_doc:
            # XXX: safeguard as long not all status messages are covered
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
            if data.startswith('<?xml'):
                print(data)
            else:
                print("%d bytes of data read" % len(data))
        self._writeln('PURGE %d' % req_id)
        self._bye()
        self.data = data
        return data

    def getWaveform(self, network, station, location, channel, starttime,
                    endtime, format="MSEED", compressed=True, getPAZ=False,
                    getCoordinates=False, route=True):
        """
        Retrieve waveform via ArcLink and returns an ObsPy Stream object.

        Parameters
        ----------
        network : string
            Network code, e.g. 'BW'.
        station : string
            Station code, e.g. 'MANZ'.
        location : string
            Location code, e.g. '01'.
        channel : string
            Channel code, e.g. 'EHE'.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        format : ['FSEED' | 'MSEED'], optional
            Output format. Either as full SEED ('FSEED') or Mini-SEED ('MSEED')
            volume (default is an 'MSEED').
            .. note::
                Format 'XSEED' is documented, but not yet implemented in
                ArcLink.
        compressed : boolean, optional
            Request compressed files from ArcLink server (default is True).
        getPAZ : boolean
            Fetch PAZ information and append to
            :class:`~obspy.core.trace.Stats` of all fetched traces. This
            considerably slows down the request.
        getCoordinates : boolean
            Fetch coordinate information and append to
            :class:`~obspy.core.trace.Stats` of all fetched traces. This
            considerably slows down the request.
        route : boolean, optional
            Enables ArcLink routing (default is True).

        Returns
        -------
            :class:`~obspy.core.stream.Stream`
        """
        tf = NamedTemporaryFile()
        self.saveWaveform(tf._fileobj, network, station, location, channel,
                          starttime, endtime, format=format,
                          compressed=compressed, route=route)
        # read stream using obspy.mseed
        tf.seek(0)
        try:
            stream = read(tf.name, 'MSEED')
        except:
            stream = Stream()
        tf.close()
        # remove temporary file:
        try:
            os.remove(tf.name)
        except:
            pass
        # trim stream
        stream.trim(starttime, endtime)
        # fetch metadata
        # fetching PAZ with wildcards: one call per channel
        if getPAZ:
            for tr in stream:
                cha = tr.stats.channel
                # XXX should add a check like metadata_check in seishub.client
                metadata = self.getMetadata(network, station, location, cha,
                                            starttime, endtime, getPAZ=getPAZ,
                                            getCoordinates=getCoordinates,
                                            route=False)
                tr.stats['paz'] = deepcopy(metadata['paz'])
        if getCoordinates:
            # reuse metadata fetched for PAZ or else fetch it
            metadata = locals().get('metadata')
            if not metadata:
                metadata = self.getMetadata(network, station, location, cha,
                                            starttime, endtime, getPAZ=getPAZ,
                                            getCoordinates=getCoordinates,
                                            route=False)
            for tr in stream:
                tr.stats['coordinates'] = deepcopy(metadata['coordinates'])
        return stream

    def saveWaveform(self, filename, network, station, location, channel,
                     starttime, endtime, format="MSEED", compressed=True,
                     route=True):
        """
        Writes a retrieved waveform directly into a file.

        This method ensures the storage of the unmodified waveform data
        delivered by the ArcLink server, e.g. preserving the record based
        quality flags of MiniSEED files which would be neglected reading it
        with obspy.mseed.

        Parameters
        ----------
        filename : string
            Name of the output file.
        network : string
            Network code, e.g. 'BW'.
        station : string
            Station code, e.g. 'MANZ'.
        location : string
            Location code, e.g. '01'.
        channel : string
            Channel code, e.g. 'EHE'.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        format : ['FSEED' | 'MSEED'], optional
            Output format. Either as full SEED ('FSEED') or Mini-SEED ('MSEED')
            volume (default is an 'MSEED').
            .. note::
                Format 'XSEED' is documented, but not yet implemented in
                ArcLink.
        compressed : boolean, optional
            Request compressed files from ArcLink server (default is True).
        route : boolean, optional
            Enables ArcLink routing (default is True).
        """
        # request type
        rtype = 'REQUEST WAVEFORM format=%s' % format
        if compressed:
            try:
                import bz2
            except:
                compressed = False
            else:
                rtype += " compression=bzip2"
        # request data
        rdata = [starttime, endtime, network, station, channel, location]
        # fetch waveform
        data = self._fetch(rtype, rdata, route=route)
        if data and compressed:
            data = bz2.decompress(data)
        # create file handler if a file name is given
        if isinstance(filename, basestring):
            fh = open(filename, "wb")
        elif isinstance(filename, file):
            fh = filename
        else:
            msg = "Parameter filename must be either string or file " + \
                "handler."
            raise TypeError(msg)
        fh.write(data)
        if isinstance(filename, basestring):
            fh.close()

    def getRouting(self, network, station, starttime, endtime):
        """
        Get responsible host addresses for given network/stations from ArcLink.

        Parameters
        ----------
        network : string
            Network code, e.g. 'BW'.
        station : string
            Station code, e.g. 'MANZ'.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.

        Returns
        -------
            Dictionary of host names.
        """
        # request type
        rtype = 'REQUEST ROUTING '
        # request data
        rdata = [starttime, endtime, network, station]
        # fetch plain XML document
        result = self._fetch(rtype, rdata, route=False)
        # parse XML document
        xml_doc = etree.fromstring(result)
        # get routing version
        if ROUTING_NS_1_0 in xml_doc.nsmap.values():
            xml_ns = ROUTING_NS_1_0
        elif ROUTING_NS_0_1 in xml_doc.nsmap.values():
            xml_ns = ROUTING_NS_0_1
        else:
            msg = "Unknown routing namespace %s"
            raise ArcLinkException(msg % xml_doc.nsmap)
        # convert into dictionary
        result = {}
        for route in xml_doc.xpath('ns0:route', namespaces={'ns0':xml_ns}):
            if xml_ns == ROUTING_NS_0_1:
                id = route.get('net_code') + '.' + route.get('sta_code')
            else:
                id = route.get('networkCode') + '.' + route.get('stationCode')
            result[id] = []
            for node in route.xpath('ns0:arclink', namespaces={'ns0':xml_ns}):
                temp = {}
                try:
                    temp['priority'] = int(node.get('priority'))
                except:
                    temp['priority'] = -1
                temp['start'] = UTCDateTime(node.get('start'))
                if node.get('end'):
                    temp['end'] = UTCDateTime(node.get('end'))
                else:
                    temp['end'] = None
                temp['host'] = node.get('address').split(':')[0].strip()
                temp['port'] = int(node.get('address').split(':')[1].strip())
                result[id].append(temp)
        return result

    def getQC(self, network, station, location, channel, starttime,
              endtime, parameters='*', outages=True, logs=True):
        """
        Retrieve QC information of ArcLink streams.

        .. note::
            Requesting QC is documented but seems not to work at the moment. 

        Parameters
        ----------
        network : string
            Network code, e.g. 'BW'.
        station : string
            Station code, e.g. 'MANZ'.
        location : string
            Location code, e.g. '01'.
        channel : string
            Channel code, e.g. 'EHE'.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        parameters : str, optional
            Comma-separated list of QC parameters. The following QC parameters
            are implemented in the present version: availability, delay, 
            gaps count, gaps interval, gaps length, latency, offset, 
            overlaps count, overlaps interval, overlaps length, rms, 
            spikes amplitude, spikes count, spikes interval, timing quality
            (default is '*' for all parameters).
        outages : boolean, optional
            Include list of outages (default is True).
        logs : boolean, optional
            Include log messages (default is True).

        Returns
        -------
            XML document as string.
        """
        # request type
        rtype = 'REQUEST QC'
        if outages is True:
            rtype += ' outages=true'
        else:
            rtype += ' outages=false'
        if logs is True:
            rtype += ' logs=false'
        else:
            rtype += ' logs=false'
        rtype += ' parameters=%s' % (parameters)
        # request data
        rdata = [starttime, endtime, network, station, channel, location]
        # fetch plain XML document
        result = self._fetch(rtype, rdata, route=False)
        return result

    def getMetadata(self, network, station, location, channel, starttime,
                    endtime, getPAZ=True, getCoordinates=True, route=True):
        """
        Returns metadata (PAZ and Coordinates).

        Parameters
        ----------
        network : string
            Network code, e.g. 'BW'.
        station : string
            Station code, e.g. 'MANZ'.
        location : string
            Location code, e.g. '01'.
        channel : string
            Channel code, e.g. 'EHE'.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        route : boolean, optional
            Enables ArcLink routing (default is True).

        Returns
        -------
            Dictionary containing keys 'paz' and 'coordinates'.
        """
        if not getPAZ and not getCoordinates:
            return {}
        result = self.getInventory(network=network, station=station,
                                   location=location, channel=channel,
                                   starttime=starttime, endtime=endtime,
                                   instruments=True, route=route)
        data = {}
        if getPAZ:
            id = '.'.join([network, station, location, channel])
            # HACK: returning first PAZ only for now
            data['paz'] = result[id][0].paz
        if getCoordinates:
            id = '.'.join([network, station])
            data['coordinates'] = AttribDict()
            for key in ['latitude', 'longitude', 'elevation']:
                data['coordinates'][key] = result[id][key]
        return data

    def __parsePAZ(self, xml_doc, xml_ns):
        """
        """
        paz = AttribDict()
        # instrument name
        paz['name'] = xml_doc.get('name', '')
        # gain
        try:
            if xml_ns == INVENTORY_NS_1_0:
                paz['gain'] = float(xml_doc.get('normalizationFactor'))
            else:
                paz['gain'] = float(xml_doc.get('norm_fac'))
        except:
            paz['gain'] = None
        # zeros
        paz['zeros'] = []
        if xml_ns == INVENTORY_NS_1_0:
            nzeros = int(xml_doc.get('numberOfZeros', 0))
        else:
            nzeros = int(xml_doc.get('nzeros', 0))
        try:
            zeros = xml_doc.xpath('ns:zeros/text()',
                                  namespaces={'ns':xml_ns})[0]
            temp = zeros.strip().replace(' ', '').replace(')(', ') (')
            for zeros in temp.split():
                paz['zeros'].append(complexifyString(zeros))
        except:
            pass
        # check number of zeros
        if len(paz['zeros']) != nzeros:
            raise ArcLinkException('Could not parse all zeros')
        # poles
        paz['poles'] = []
        if xml_ns == INVENTORY_NS_1_0:
            npoles = int(xml_doc.get('numberOfPoles', 0))
        else:
            npoles = int(xml_doc.get('npoles', 0))
        try:
            poles = xml_doc.xpath('ns:poles/text()',
                                  namespaces={'ns':xml_ns})[0]
            temp = poles.strip().replace(' ', '').replace(')(', ') (')
            for poles in temp.split():
                paz['poles'].append(complexifyString(poles))
        except:
            pass
        # check number of poles
        if len(paz['poles']) != npoles:
            raise ArcLinkException('Could not parse all poles')
        return paz

    def getPAZ(self, network, station, location, channel, starttime, endtime):
        """
        Returns poles, zeros, gain and sensitivity of a single channel.

        Parameters
        ----------
        network : string
            Network code, e.g. 'BW'.
        station : string
            Station code, e.g. 'MANZ'.
        location : string
            Location code, e.g. '01'.
        channel : string
            Channel code, e.g. 'EHE'.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.

        Returns
        -------
            Dictionary containing PAZ information.
        """
        result = self.getInventory(network=network, station=station,
                                   location=location, channel=channel,
                                   starttime=starttime, endtime=endtime,
                                   instruments=True)
        id = '.'.join([network, station, location, channel])
        if '*' in id:
            msg = 'getPAZ supports only a single channel, use getInventory' + \
                  ' instead'
            raise ArcLinkException(msg)
        try:
            # XXX: why dict of instruments? Only one instrument is returned!
            # HACK: returning first PAZ only for now
            paz = result[id][0].paz
            return {paz.name: paz}
        except:
            msg = 'Could not find PAZ for channel %s' % id
            raise ArcLinkException(msg)

    def saveResponse(self, filename, network, station, location, channel,
                     starttime, endtime, format='SEED'):
        """
        Writes response information into a file.

        Parameters
        ----------
        filename : string
            Name of the output file.
        network : string
            Network code, e.g. 'BW'.
        station : string
            Station code, e.g. 'MANZ'.
        location : string
            Location code, e.g. '01'.
        channel : string
            Channel code, e.g. 'EHE'.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        format : ['SEED'], optional
            Output format. Currently only Dataless SEED is supported.
        """
        # request type
        rtype = 'REQUEST RESPONSE format=%s' % format
        # request data
        rdata = [starttime, endtime, network, station, channel, location]
        # fetch dataless
        data = self._fetch(rtype, rdata)
        fh = open(filename, "wb")
        fh.write(data)
        fh.close()

    def getInventory(self, network, station='*', location='*', channel='*',
                     starttime=UTCDateTime(), endtime=UTCDateTime(),
                     instruments=False, route=True, sensortype='',
                     min_latitude=None, max_latitude=None,
                     min_longitude=None, max_longitude=None,
                     restricted=None, permanent=None):
        """
        Returns inventory data.

        Parameters
        ----------
        network : string
            Network code, e.g. 'BW'.
        station : string
            Station code, e.g. 'MANZ'.
        location : string
            Location code, e.g. '01'.
        channel : string
            Channel code, e.g. 'EHE'.
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        instruments : boolean, optional
            Include instrument data (default is False).
        route : boolean, optional
            Enables ArcLink routing (default is True).
        sensortype : string, optional
            Limit streams to those using specific sensor types: "VBB", 
            "BB", "SM", "OBS", etc. Can be also a combination like "VBB+BB+SM".
        min_latitude : float, optional
            Minimum latitude
        max_latitude : float, optional
            Maximum latitude
        min_longitude : float, optional
            Minimum longitude
        max_longitude : float, optional
            Maximum longitude
        permanent : boolean, optional
            Requesting only permanent or temporary networks respectively.
            Default is None, therefore requesting all data.
        restricted : boolean, optional
            Requesting only networks/stations/streams that have restricted or
            open data respectively. Default is None.

        Returns
        -------
            Dictionary of inventory information.
        """
        # request type
        rtype = 'REQUEST INVENTORY '
        if instruments:
            rtype += 'instruments=true '
        # request data
        rdata = [starttime, endtime, network, station, channel, location, '.']
        if restricted is True:
            rdata.append('restricted=true')
        elif restricted is False:
            rdata.append('restricted=false')
        if permanent is True:
            rdata.append('permanent=true')
        elif permanent is False:
            rdata.append('permanent=false')
        if sensortype != '':
            rdata.append('sensortype=%s' % sensortype)
        if min_latitude:
            rdata.append('latmin=%f' % min_latitude)
        if max_latitude:
            rdata.append('latmax=%f' % max_latitude)
        if min_longitude:
            rdata.append('lonmin=%f' % min_longitude)
        if max_longitude:
            rdata.append('lonmax=%f' % max_longitude)
        # fetch plain XML document
        if network == '*':
            # set route to False if not network id is given
            result = self._fetch(rtype, rdata, route=False)
        else:
            result = self._fetch(rtype, rdata, route=route)
        # parse XML document
        xml_doc = etree.fromstring(result)
        # get routing version
        if INVENTORY_NS_1_0 in xml_doc.nsmap.values():
            xml_ns = INVENTORY_NS_1_0
            stream_ns = 'sensorLocation'
            component_ns = 'stream'
            seismometer_ns = 'sensor'
            name_ns = 'publicID'
            resp_paz_ns = 'responsePAZ'
        elif INVENTORY_NS_0_2 in xml_doc.nsmap.values():
            xml_ns = INVENTORY_NS_0_2
            stream_ns = 'seis_stream'
            component_ns = 'component'
            seismometer_ns = 'seismometer'
            name_ns = 'name'
            resp_paz_ns = 'resp_paz'
        else:
            msg = "Unknown inventory namespace %s"
            raise ArcLinkException(msg % xml_doc.nsmap)
        # convert into dictionary
        data = AttribDict()
        for network in xml_doc.xpath('ns:network', namespaces={'ns':xml_ns}):
            net = AttribDict()
            # strings
            for key in ['archive', 'code', 'description', 'institutions',
                        'net_class', 'region', 'type']:
                net[key] = network.get(key, '')
            # restricted
            if network.get('restricted', '') == 'false':
                net['restricted'] = False
            else:
                net['restricted'] = True
            # date / times
            try:
                net.start = UTCDateTime(network.get('start'))
            except:
                net.start = None
            try:
                net.end = UTCDateTime(network.get('end'))
            except:
                net.end = None
            # remark
            try:
                net.remark = network.xpath('ns:remark',
                    namespaces={'ns':xml_ns})[0].text or ''
            except:
                net.remark = ''
            # write network entries
            data[net.code] = net
            # stations
            for station in network.xpath('ns0:station',
                                         namespaces={'ns0':xml_ns}):
                sta = AttribDict()
                # strings
                for key in ['code', 'description', 'affiliation', 'country',
                            'place', 'restricted', 'archive_net']:
                    sta[key] = station.get(key, '')
                # floats
                for key in ['elevation', 'longitude', 'depth', 'latitude']:
                    try:
                        sta[key] = float(station.get(key))
                    except:
                        sta[key] = None
                # restricted
                if station.get('restricted', '') == 'false':
                    sta['restricted'] = False
                else:
                    sta['restricted'] = True
                # date / times
                try:
                    sta.start = UTCDateTime(station.get('start'))
                except:
                    sta.start = None
                try:
                    sta.end = UTCDateTime(station.get('end'))
                except:
                    sta.end = None
                # remark
                try:
                    sta.remark = station.xpath('ns:remark',
                        namespaces={'ns':xml_ns})[0].text or ''
                except:
                    sta.remark = ''
                # write station entry
                data[net.code + '.' + sta.code] = sta
                # instruments
                for stream in station.xpath('ns:' + stream_ns,
                                            namespaces={'ns':xml_ns}):
                    # date / times
                    try:
                        start = UTCDateTime(stream.get('start'))
                    except:
                        start = None
                    try:
                        end = UTCDateTime(stream.get('end'))
                    except:
                        end = None
                    # check date/time boundaries
                    if start > endtime:
                        continue
                    if end and starttime > end:
                        continue
                    # fetch component
                    for comp in stream.xpath('ns:' + component_ns,
                                             namespaces={'ns':xml_ns}):
                        if xml_ns == INVENTORY_NS_0_2:
                            seismometer_id = stream.get(seismometer_ns, None)
                        else:
                            seismometer_id = comp.get(seismometer_ns, None)
                        # channel id
                        if xml_ns == INVENTORY_NS_0_2:
                            # channel code is split into two attributes
                            id = '.'.join([net.code, sta.code,
                                           stream.get('loc_code' , ''),
                                           stream.get('code' , '  ') + \
                                           comp.get('code', ' ')])
                        else:
                            id = '.'.join([net.code, sta.code,
                                           stream.get('code' , ''),
                                           comp.get('code', '')])
                        # write channel entry
                        if not id in data:
                            data[id] = []
                        temp = AttribDict()
                        data[id].append(temp)
                        # fetch gain
                        try:
                            temp['gain'] = float(comp.get('gain'))
                        except:
                            temp['gain'] = None
                        # date / times
                        try:
                            temp['starttime'] = UTCDateTime(comp.get('start'))
                        except:
                            temp['starttime'] = None
                        try:
                            temp['endtime'] = UTCDateTime(comp.get('end'))
                        except:
                            temp['endtime'] = None
                        if not instruments or not seismometer_id:
                            continue
                        # PAZ
                        paz_id = xml_doc.xpath('ns:' + seismometer_ns + \
                                               '[@' + name_ns + '="' + \
                                               seismometer_id + '"]/@response',
                                               namespaces={'ns':xml_ns})
                        if not paz_id:
                            continue
                        paz_id = paz_id[0]
                        # hack for 0.2 schema
                        if paz_id.startswith('paz:'):
                            paz_id = paz_id[4:]
                        xml_paz = xml_doc.xpath('ns:' + resp_paz_ns + '[@' + \
                                                name_ns + '="' + paz_id + '"]',
                                                namespaces={'ns':xml_ns})
                        if not xml_paz:
                            continue
                        # parse PAZ
                        paz = self.__parsePAZ(xml_paz[0], xml_ns)
                        # sensitivity
                        # here we try to overwrites PAZ with component gain
                        try:
                            paz['sensitivity'] = float(comp.get('gain'))
                        except:
                            paz['sensitivity'] = paz['gain']
                        temp['paz'] = paz
        return data

    def getNetworks(self, starttime, endtime):
        """
        Returns a dictionary of available networks within the given time span.

        .. note::
            Currently the time span seems to be ignored by the ArcLink servers,
            therefore all possible networks are returned.

        Parameters
        ----------
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.

        Returns
        -------
            Dictionary of network data.
        """
        return self.getInventory(network='*', starttime=starttime,
                                 endtime=endtime, route=False)

    def getStations(self, starttime, endtime, network):
        """
        Returns a dictionary of available stations in the given network(s).

        .. note::
            Currently the time span seems to be ignored by the ArcLink servers,
            therefore all possible stations are returned.

        Parameters
        ----------
        starttime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            Start date and time.
        endtime : :class:`~obspy.core.utcdatetime.UTCDateTime`
            End date and time.
        network : string
            Network code, e.g. 'BW'.

        Returns
        -------
            Dictionary of station data.
        """
        data = self.getInventory(network=network, starttime=starttime,
                                 endtime=endtime)
        stations = [value for key, value in data.items() \
                    if key.startswith(network + '.') \
                    and "code" in value]
        return stations


if __name__ == '__main__': # pragma: no cover
    import doctest
    doctest.testmod(exclude_empty=True)
