# -*- coding: utf-8 -*-
"""
ArcLink client.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from copy import deepcopy
from lxml import objectify, etree
from obspy.core import read, Stream, UTCDateTime
from obspy.core.util import NamedTemporaryFile, AttribDict, complexifyString, \
    deprecated
from telnetlib import Telnet
import os
import sys
import time
import warnings


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

    Notes
    -----
    The following ArcLink servers may be accessed via ObsPy:

    Public servers:
      * WebDC servers: webdc.eu:18001, webdc:18002

    Further mirrors listed at webdc.eu (restricted access only):
      * ODC Server:  bhlsa03.knmi.nl:18001
      * INGV Server: eida.rm.ingv.it:18001
      * IPGP Server: geosrt2.ipgp.fr:18001
    """
    status_timeout = 2
    status_delay = 0.1

    def __init__(self, host="webdc.eu", port=18002, timeout=20,
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

    def _fetch(self, request_type, request_data, route=True):
        # skip routing on request
        if not route:
            return self._request(request_type, request_data)
        # using route
        routes = self.getRouting(network_id=request_data[2],
                                 station_id=request_data[3],
                                 start_datetime=request_data[0],
                                 end_datetime=request_data[1])
        id = request_data[2] + '.' + request_data[3]
        if id in routes.keys() and routes[id] == []:
            # we are at the responsible ArcLink node
            return self._request(request_type, request_data)
        id = request_data[2] + '.'
        if id not in routes.keys():
            msg = 'Could not find route to %s.%s'
            raise ArcLinkException(msg % (request_data[2], request_data[3]))
        routes = routes[id]
        routes.sort(lambda x, y: cmp(x['priority'], y['priority']))
        for route in routes:
            self.host = route['host']
            self.port = route['port']
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
        # create request string
        out = (request_data[0] - 1).formatArcLink() + ' '
        out += (request_data[1] + 1).formatArcLink() + ' '
        out += ' '.join([str(i) for i in request_data[2:]])
        self._writeln(out)
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
        if 'status="NODATA"' in xml_doc:
            # no data
            self._writeln('PURGE %d' % req_id)
            self._bye()
            raise ArcLinkException('No data (e.g. wrong route)')
        elif 'id="NODATA"' in xml_doc or 'id="ERROR"' in xml_doc:
            # error or no data
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
                print data
            else:
                print "%d bytes of data read" % len(data)
        self._writeln('PURGE %d' % req_id)
        self._bye()
        self.data = data
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
        rdata = [start_datetime, end_datetime, network_id, station_id,
                 channel_id, location_id]
        data = self._fetch(rtype, rdata)
        if data and compressed:
            data = bz2.decompress(data)
        # create file handler if a file name is given
        if isinstance(filename, basestring):
            fh = open(filename, "wb")
        else:
            fh = filename
        fh.write(data)
        if isinstance(filename, basestring):
            fh.close()

    def getWaveform(self, network_id, station_id, location_id, channel_id,
                    start_datetime, end_datetime, format="MSEED",
                    compressed=True, getPAZ=False, getCoordinates=False):
        """
        Retrieve waveform via ArcLink and returns an ObsPy Stream object.

        :type filename: string
        :param filename: Name of the output file.
        :type network_id: string
        :param network_id: Network code, e.g. 'BW'.
        :type station_id: string
        :param station_id: Station code, e.g. 'MANZ'.
        :type location_id: string
        :param location_id: Location code, e.g. '01'.
        :type channel_id: string
        :param channel_id: Channel code, e.g. 'EHE'.
        :type start_datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param start_datetime: Start date and time.
        :type end_datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param end_datetime: End date and time.
        :type format: String
        :param format: Output format. Either full SEED ('FSEED') or Mini-SEED
                ('MSEED') volume (default is 'MSEED'). 
                Note: Format 'XSEED' is documented, but not yet implemented in
                ArcLink.
        :type compressed: boolean, optional 
        :paramcompressed: Request compressed files from ArcLink server (default
                is True).
        :type getPAZ: Boolean
        :param getPAZ: Fetch PAZ information and append to
            :class:`~obspy.core.trace.Stats` of all fetched traces. This
            considerably slows down the request.
        :type getCoordinates: Boolean
        :param getCoordinates: Fetch coordinate information and append to
            :class:`~obspy.core.trace.Stats` of all fetched traces. This
            considerably slows down the request.

        :returns: :class:`~obspy.core.stream.Stream`
        """
        tf = NamedTemporaryFile()
        self.saveWaveform(tf, network_id, station_id, location_id, channel_id,
                          start_datetime, end_datetime, format=format,
                          compressed=compressed)
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
        stream.trim(start_datetime, end_datetime)
        # fetch metadata
        # fetching PAZ with wildcards: one call per channel
        if getPAZ:
            for tr in stream:
                cha = tr.stats.channel
                # XXX should add a check like metadata_check in seishub.client
                metadata = self.getMetadata(network_id, station_id,
                                            location_id, cha, start_datetime,
                                            end_datetime, getPAZ=getPAZ,
                                            getCoordinates=getCoordinates)
                tr.stats['paz'] = deepcopy(metadata['paz'])
        if getCoordinates:
            # reuse metadata fetched for paz or else fetch it
            metadata = locals().get('metadata')
            if not metadata:
                metadata = self.getMetadata(network_id, station_id,
                                            location_id, channel_id,
                                            start_datetime, end_datetime,
                                            getPAZ=getPAZ,
                                            getCoordinates=getCoordinates)
            for tr in stream:
                tr.stats['coordinates'] = deepcopy(metadata['coordinates'])
        return stream

    def getRouting(self, network_id, station_id, start_datetime, end_datetime):
        """
        Get responsible host addresses for given network/stations from ArcLink.

        :type network_id: string
        :param network_id: Network code, e.g. 'BW'.
        :type station_id: string
        :param station_id: Station code, e.g. 'MANZ'.
        :type start_datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param start_datetime: Start date and time.
        :type end_datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param end_datetime: End date and time.

        :returns: dict of host names.
        """
        rtype = 'REQUEST ROUTING '
        # adding one second to start and end time to ensure right date times
        rdata = [start_datetime, end_datetime, network_id, station_id]
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
                temp['priority'] = int(node.get('priority'))
                temp['start'] = UTCDateTime(node.get('start'))
                if node.get('end'):
                    temp['end'] = UTCDateTime(node.get('end'))
                else:
                    temp['end'] = None
                temp['host'] = node.get('address').split(':')[0].strip()
                temp['port'] = int(node.get('address').split(':')[1].strip())
                result[id].append(temp)
        return result

    def getMetadata(self, network_id, station_id, location_id, channel_id,
                    start_datetime, end_datetime, getPAZ=True,
                    getCoordinates=True):
        """
        Returns metadata (PAZ and Coordinates).

        :type network_id: String
        :param network_id: Network code, e.g. 'BW'.
        :type station_id: String
        :param station_id: Station code, e.g. 'MANZ'.
        :type location_id: String
        :param location_id: Location code, e.g. '01'.
        :type channel_id: String
        :param channel_id: Channel code, e.g. 'EHE', "*" for component allowed
        :param start_datetime: start time as L{obspy.UTCDateTime} object.
        :param end_datetime: end time as L{obspy.UTCDateTime} object.
        :return: dictionary containing keys 'paz' and 'coordinates'
        """
        if not getPAZ and not getCoordinates:
            return {}
        result = self.getInventory(network_id=network_id,
                                   station_id=station_id,
                                   location_id=location_id,
                                   channel_id=channel_id,
                                   start_datetime=start_datetime,
                                   end_datetime=end_datetime,
                                   instruments=True)
        data = {}
        if getPAZ:
            id = '.'.join([network_id, station_id, location_id, channel_id])
            data['paz'] = result[id].paz
        if getCoordinates:
            id = '.'.join([network_id, station_id])
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

    def getPAZ(self, network_id, station_id, location_id, channel_id,
               start_datetime, end_datetime):
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
        result = self.getInventory(network_id=network_id,
                                   station_id=station_id,
                                   location_id=location_id,
                                   channel_id=channel_id,
                                   start_datetime=start_datetime,
                                   end_datetime=end_datetime,
                                   instruments=True)
        id = '.'.join([network_id, station_id, location_id, channel_id])
        paz = result[id].paz
        # XXX: why dict of instruments? Only one instrument is returned!
        return {paz.name: paz}

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
        rdata = [start_datetime, end_datetime, network_id, station_id,
                 channel_id, location_id]
        data = self._fetch(rtype, rdata)
        fh = open(filename, "wb")
        fh.write(data)
        fh.close()

    def getInventory(self, network_id, station_id='*', location_id='*',
                     channel_id='*', start_datetime=UTCDateTime(),
                     end_datetime=UTCDateTime(), instruments=False,
                     route=True):
        """
        """
        rtype = 'REQUEST INVENTORY '
        if instruments:
            rtype += 'instruments=true '
        # adding one second to start and end time to ensure right date times
        rdata = [start_datetime, end_datetime, network_id, station_id,
                 channel_id, location_id]
        # fetch plain XML document
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
            temp = AttribDict()
            # strings
            for key in ['archive', 'code', 'description', 'institutions',
                        'net_class', 'region', 'type']:
                temp[key] = network.get(key, '')
            # restricted
            if network.get('restricted', '') == 'false':
                temp['restricted'] = False
            else:
                temp['restricted'] = True
            # date / times
            try:
                temp.start = UTCDateTime(network.get('start'))
            except:
                temp.start = None
            try:
                temp.end = UTCDateTime(network.get('end'))
            except:
                temp.end = None
            # remark
            try:
                temp.remark = network.xpath('ns:remark',
                    namespaces={'ns':xml_ns})[0].text or ''
            except:
                temp.remark = ''
            # write network entries
            data[temp.code] = temp
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
                data[temp.code + '.' + sta.code] = sta
                if not instruments:
                    continue
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
                    if end and end_datetime > end:
                        continue
                    if start_datetime < start:
                        continue
                    # fetch component
                    for comp in stream.xpath('ns:' + component_ns,
                                             namespaces={'ns':xml_ns}):
                        if xml_ns == INVENTORY_NS_0_2:
                            seismometer_id = stream.get(seismometer_ns, None)
                        else:
                            seismometer_id = comp.get(seismometer_ns, None)
                        if not seismometer_id:
                            continue
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
                        id = temp.code + '.' + sta.code + '.' + \
                            stream.get('loc_code' , '') + '.' + \
                            stream.get('code' , '  ') + \
                            comp.get('code', ' ').strip()
                        # parse PAZ
                        paz = self.__parsePAZ(xml_paz[0], xml_ns)
                        # sensitivity
                        # here we try to overwrites PAZ with component gain 
                        try:
                            paz['sensitivity'] = float(comp.get('gain'))
                        except:
                            paz['sensitivity'] = paz['gain']
                        # write channel entry
                        data[id] = AttribDict()
                        data[id]['paz'] = paz
        return data

    @deprecated
    def getNetworks(self, start_datetime, end_datetime):
        """
        Returns a dictionary of available networks within the given time span.

        Currently the time span is ignored by the ArcLink servers, therefore
        all possible networks are returned.

        :param start_datetime: start time as L{obspy.UTCDateTime} object.
        :param end_datetime: end time as L{obspy.UTCDateTime} object.
        :return: dictionary of network data.
        """
        return self.getInventory(network_id='*',
                                 start_datetime=start_datetime,
                                 end_datetime=end_datetime, route=False)

    @deprecated
    def getStations(self, start_datetime, end_datetime, network_id):
        """
        Returns a dictionary of available stations in the given network(s).

        Currently the time span is ignored by the ArcLink servers, therefore
        all possible stations are returned.

        :type start_datetime: :class:`obspy.core.util.UTCDateTime`
        :param start_datetime: start time
        :type end_datetime: :class:`obspy.core.util.UTCDateTime`
        :param end_datetime: end time
        :type network_id: String or list of Strings
        :param network_id: Network(s) to list stations of
        :return: dictionary of station data.
        """
        data = self.getInventory(network_id=network_id,
                                 start_datetime=start_datetime,
                                 end_datetime=end_datetime)
        return [value for key, value in data.items() \
                if key.startswith(network_id + '.')]
