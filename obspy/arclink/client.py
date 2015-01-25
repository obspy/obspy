# -*- coding: utf-8 -*-
"""
ArcLink/WebDC client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import native_str

from obspy import read, UTCDateTime
from obspy.core.util import AttribDict, complexifyString
from obspy.core.util.decorator import deprecated_keywords

from fnmatch import fnmatch
import io
from lxml import objectify, etree
from telnetlib import Telnet
from numpy import pi
import os
import time
import warnings

DCID_KEY_FILE = os.path.join(os.getenv('HOME') or '', 'dcidpasswords.txt')
MAX_REQUESTS = 50

_ROUTING_NS_1_0 = "http://geofon.gfz-potsdam.de/ns/Routing/1.0/"
_ROUTING_NS_0_1 = "http://geofon.gfz-potsdam.de/ns/routing/0.1/"
_INVENTORY_NS_1_0 = "http://geofon.gfz-potsdam.de/ns/Inventory/1.0/"
_INVENTORY_NS_0_2 = "http://geofon.gfz-potsdam.de/ns/inventory/0.2/"

MSG_NOPAZ = "No Poles and Zeros information returned by server."

MSG_USER_REQUIRED = """Initializing a ArcLink client without the user keyword
is deprecated! Please provide a proper user identification string such as your
email address. Defaulting to 'ObsPy client' for now."""


class ArcLinkException(Exception):
    """
    Raised by the ArcLink client for known exceptions.
    """


class Client(object):
    """
    The ArcLink/WebDC client.

    :type host: str, optional
    :param host: Host name of the remote ArcLink server (default host is
        ``'webdc.eu'``).
    :type port: int, optional
    :param port: Port of the remote ArcLink server (default port is ``18002``).
    :type timeout: int, optional
    :param timeout: Seconds before a connection timeout is raised (default is
        ``20`` seconds).
    :type user: str
    :param user: The user name is used for identification with the ArcLink
        server. This entry is also used for usage statistics within the data
        centers, so please provide a meaningful user id such as your email
        address.
    :type password: str, optional
    :param password: A password used for authentication with the ArcLink server
        (default is an empty string).
    :type institution: str, optional
    :param institution: A string containing the name of the institution of the
        requesting person (default is an ``'Anonymous'``).
    :type dcid_keys: dict, optional
    :param dcid_keys: Dictionary of data center ids (DCID) and passwords used
        for decoding encrypted waveform requests.
    :type dcid_key_file: str, optional
    :param dcid_key_file: Simple text configuration file containing lines of
        data center ids (DCIDs) and password pairs separated by a equal sign,
        e.g. for DCID ``BIA`` and password ``OfH9ekhi`` use ``"BIA=OfH9ekhi"``.
        If not set, passwords found in a file called `$HOME/dcidpasswords.txt`
        will be used automatically.
    :type debug: bool, optional
    :param debug: Enables verbose output of the connection handling (default is
        ``False``).
    :type command_delay: float, optional
    :param command_delay: Delay between each command send to the ArcLink server
        (default is ``0``).
    :type status_delay: float, optional
    :param status_delay: Delay in seconds between each status request (default
        is ``0.5`` seconds).

    .. rubric:: Notes

    The following ArcLink servers may be accessed (also see
    http://www.orfeus-eu.org/eida/eida_advanced_users.html;
    maybe partly restricted access only):

    * WebDC: webdc.eu:18001, webdc.eu:18002
    * ODC:   eida.knmi.nl:18002
    * GFZ:   eida.gfz-potsdam.de:18001
    * RESIF: eida.resif.fr:18001
    * INGV:  --
    * ETHZ:  eida.ethz.ch:18001
    * BGR:   eida.bgr.de:18001
    * IPGP:  eida.ipgp.fr:18001
    * USP:   seisrequest.iag.usp.br:18001
    """
    def __init__(self, host="webdc.eu", port=18002, user=None,
                 password="", institution="Anonymous", timeout=20,
                 dcid_keys={}, dcid_key_file=None, debug=False,
                 command_delay=0, status_delay=0.5):
        """
        Initializes an ArcLink client.

        See :class:`obspy.arclink.client.Client` for all parameters.
        """
        if user is None:
            warnings.warn(MSG_USER_REQUIRED, category=DeprecationWarning)
            self.user = 'ObsPy client'
        else:
            self.user = user
        self.password = password
        self.institution = institution
        self.command_delay = command_delay
        self.status_delay = status_delay

        self.init_host = host
        self.init_port = port
        self.timeout = timeout
        self.dcid_keys = dcid_keys
        self._client = Telnet(host, port, timeout)
        # silent connection check
        self.debug = False
        self._hello()
        self.debug = debug
        if self.debug:
            print('\nConnected to %s:%s' % (self._client.host,
                                            str(self._client.port)))
        # check for dcid_key_file
        if not dcid_key_file:
            # check in user directory
            if not os.path.isfile(DCID_KEY_FILE):
                return
            dcid_key_file = DCID_KEY_FILE
        # parse dcid_key_file
        try:
            with open(dcid_key_file, 'rt') as fp:
                lines = fp.readlines()
        except:
            pass
        else:
            for line in lines:
                line = line.strip()
                # skip empty lines
                if not line:
                    continue
                # skip comments
                if line.startswith('#'):
                    continue
                if ' ' in line:
                    key, value = line.split(' ', 1)
                else:
                    key, value = line.split('=', 1)
                key = key.strip()
                # ensure that dcid_keys set via parameters are not overwritten
                if key not in self.dcid_keys:
                    self.dcid_keys[key] = value.strip()

    def _reconnect(self):
        self._client.close()
        try:
            self._client.open(native_str(self._client.host),
                              self._client.port,
                              self._client.timeout)
        except:
            # Python 2.6: port needs to be native int or string -> not long
            self._client.open(native_str(self._client.host),
                              native_str(self._client.port),
                              self._client.timeout)

    def _writeln(self, buffer):
        # Py3k: might be confusing, _writeln accepts str
        # readln accepts bytes (but was smallest change like that)
        if self.command_delay:
            time.sleep(self.command_delay)
        b_buffer = (buffer + '\r\n').encode()
        self._client.write(b_buffer)
        if self.debug:
            print(b'>>> ' + b_buffer)

    def _readln(self, value=b''):
        line = self._client.read_until(value + b'\r\n', self.timeout)
        line = line.strip()
        if value not in line:
            msg = "Timeout waiting for expected %s, got %s"
            raise ArcLinkException(msg % (value, line.decode()))
        if self.debug:
            print(b'... ' + line)
        return line

    def _hello(self):
        self._reconnect()
        self._writeln('HELLO')
        self.version = self._readln(b')')
        self.node = self._readln()
        if self.password:
            self._writeln('USER %s %s' % (self.user, self.password))
        else:
            self._writeln('USER %s' % self.user)
        self._readln(b'OK')
        self._writeln('INSTITUTION %s' % self.institution)
        self._readln(b'OK')

    def _bye(self):
        self._writeln('BYE')
        self._client.close()

    def _fetch(self, request_type, request_data, route=True):
        # skip routing on request
        if not route:
            # always use initial node if routing is disabled
            self._client.host = self.init_host
            self._client.port = self.init_port
            return self._request(request_type, request_data)
        # request routing table for given network/station/times combination
        # location and channel information are ignored by ArcLink
        routes = self.getRouting(network=request_data[2],
                                 station=request_data[3],
                                 starttime=request_data[0],
                                 endtime=request_data[1])
        # search routes for network/station/location/channel
        table = self._findRoute(routes, request_data)
        if not table:
            # retry first ArcLink node if host or port has been changed
            if self._client.host != self.init_host or \
               self._client.port != self.init_port:
                self._client.host = self.init_host
                self._client.port = self.init_port
                if self.debug:
                    print('\nRequesting %s:%d' % (self._client.host,
                                                  self._client.port))
                return self._fetch(request_type, request_data, route)
            msg = 'Could not find route to %s.%s. If you think the data ' + \
                  'should be there, you might want to retry ' + \
                  'with manually connecting to a different ArcLink node ' + \
                  '(see docstring of Client) to see if there is a problem ' + \
                  'with a routing table at a specific ArcLink node (and ' + \
                  'contact the ArcLink node operators).'
            raise ArcLinkException(msg % (request_data[2], request_data[3]))
        # we got a routing table
        for item in table:
            if item == {}:
                return self._request(request_type, request_data)
            # check if current connection is enough
            if item['host'] == self._client.host and \
               item['port'] == self._client.port:
                return self._request(request_type, request_data)
            self._client.host = item['host']
            self._client.port = item['port']
            if self.debug:
                print('\nRequesting %s:%d' % (self._client.host,
                                              self._client.port))
            self._reconnect()
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
        # adding one second to start and end time to ensure right date times
        out = (request_data[0] - 1).formatArcLink() + ' '
        out += (request_data[1] + 1).formatArcLink() + ' '
        out += ' '.join([str(i) for i in request_data[2:]])
        self._writeln(out)
        self._writeln('END')
        self._readln(b'OK')
        # get status id
        while True:
            status = self._readln()
            try:
                req_id = int(status)
            except:
                if 'ERROR' in status:
                    self._bye()
                    raise ArcLinkException('Error requesting status id')
                pass
            else:
                break
        # loop until we hit ready="true" in the status message
        _loops = 0
        _old_xml_doc = None
        while True:
            self._writeln('STATUS %d' % req_id)
            xml_doc = self._readln(b'END')
            if b'ready="true"' in xml_doc:
                break
            # check if status messages changes over time
            if _old_xml_doc == xml_doc:
                _loops += 1
            else:
                _loops = 0
                _old_xml_doc = xml_doc
            # if we hit MAX_REQUESTS equal status break the loop
            if _loops > MAX_REQUESTS:
                msg = 'MAX_REQUESTS exceeded - breaking current request loop'
                warnings.warn(msg, UserWarning)
                break
            # wait a bit
            time.sleep(self.status_delay)
        # check for errors
        for err_code in (b'DENIED', b'CANCELLED', b'CANCEL', b'ERROR',
                         b'RETRY', b'WARN', b'UNSET'):
            err_str = b'status="' + err_code + b'"'
            if err_str in xml_doc:
                # cleanup
                self._writeln('PURGE %d' % req_id)
                self._bye()
                # parse XML for reason
                xml_doc = objectify.fromstring(xml_doc[:-3])
                msg = xml_doc.request.volume.line.get('message')
                raise ArcLinkException("%s %s" % (err_code, msg))
        if b'status="NODATA"' in xml_doc:
            # cleanup
            self._writeln('PURGE %d' % req_id)
            self._bye()
            raise ArcLinkException('No data available')
        elif b'id="NODATA"' in xml_doc or b'id="ERROR"' in xml_doc:
            # cleanup
            self._writeln('PURGE %d' % req_id)
            self._bye()
            # parse XML for error message
            xml_doc = objectify.fromstring(xml_doc[:-3])
            raise ArcLinkException(xml_doc.request.volume.line.get('message'))
        elif b'<line content' not in xml_doc:
            # safeguard for not covered status messages
            self._writeln('PURGE %d' % req_id)
            self._bye()
            msg = "Uncovered status message - contact a developer to fix this"
            raise ArcLinkException(msg)
        self._writeln('DOWNLOAD %d' % req_id)
        try:
            fd = self._client.get_socket().makefile('rb')
            length = int(fd.readline(100).strip())
            data = b''
            while len(data) < length:
                buf = fd.read(min(4096, length - len(data)))
                data += buf
            buf = fd.readline(100).strip()
            if buf != b"END" or len(data) != length:
                raise Exception('Wrong length!')
            if self.debug:
                if data.startswith(b'<?xml'):
                    print(data)
                else:
                    print("%d bytes of data read" % len(data))
        finally:
            self._writeln('PURGE %d' % req_id)
            self._bye()
        # check for encryption
        if b'encrypted="true"' in xml_doc:
            # extract dcid
            xml_doc = objectify.fromstring(xml_doc[:-3])
            dcid = xml_doc.request.volume.get('dcid')
            # check if given in known list of keys
            if dcid in self.dcid_keys:
                # call decrypt routine
                from obspy.arclink.decrypt import SSLWrapper
                decryptor = SSLWrapper(self.dcid_keys[dcid])
                data = decryptor.update(data)
                data += decryptor.final()
            else:
                msg = "Could not decrypt waveform data for dcid %s."
                warnings.warn(msg % (dcid))
        return data

    def getWaveform(self, network, station, location, channel, starttime,
                    endtime, format="MSEED", compressed=True, metadata=False,
                    route=True, **kwargs):
        """
        Retrieves waveform data via ArcLink and returns an ObsPy Stream object.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :type location: str
        :param location: Location code, e.g. ``'01'``. Location code may
            contain wild cards.
        :type channel: str
        :param channel: Channel code, e.g. ``'EHE'``. Channel code may
            contain wild cards.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type format: str, optional
        :param format: Output format. Either as full SEED (``'FSEED'``) or
            Mini-SEED (``'MSEED'``) volume. Defaults to ``'MSEED'``.
        :type compressed: bool, optional
        :param compressed: Request compressed files from ArcLink server.
            Defaults to ``True``.
        :type metadata: bool, optional
        :param metadata: Fetch PAZ and coordinate information and append to
            :class:`~obspy.core.trace.Stats` of all fetched traces. Defaults
            to ``False``.
        :type route: bool, optional
        :param route: Enables ArcLink routing. Defaults to ``True``.
        :return: ObsPy :class:`~obspy.core.stream.Stream` object.

        .. rubric:: Example

        >>> from obspy.arclink import Client
        >>> from obspy import UTCDateTime
        >>> client = Client("webdc.eu", 18001, user='test@obspy.org')
        >>> t = UTCDateTime("2009-08-20 04:03:12")
        >>> st = client.getWaveform("BW", "RJOB", "", "EH*", t - 3, t + 15)
        >>> st.plot() #doctest: +SKIP

        .. plot::

            from obspy import UTCDateTime
            from obspy.arclink.client import Client
            client = Client("webdc.eu", 18001, 'test@obspy.org')
            t = UTCDateTime("2009-08-20 04:03:12")
            st = client.getWaveform("BW", "RJOB", "", "EH*", t - 3, t + 15)
            st.plot()
        """
        if kwargs.get('getPAZ') or kwargs.get('getCoordinates'):
            msg = "Keywords getPAZ and getCoordinates are deprecated. " + \
                  "Please use keyword metadata instead."
            warnings.warn(msg, DeprecationWarning)
        # handle deprecated keywords - one must be True to enable metadata
        metadata = metadata or kwargs.get('getPAZ', False) or \
            kwargs.get('getCoordinates', False)
        file_stream = io.BytesIO()
        self.saveWaveform(file_stream, network, station, location, channel,
                          starttime, endtime, format=format,
                          compressed=compressed, route=route)
        file_stream.seek(0, 0)
        stream = read(file_stream, 'MSEED')
        file_stream.close()
        # trim stream
        stream.trim(starttime, endtime)
        # fetching PAZ and coordinates
        if metadata:
            # fetch metadata only once
            inv = self.getInventory(network=network, station=station,
                                    location=location, channel=channel,
                                    starttime=starttime, endtime=endtime,
                                    instruments=True, route=False)
            netsta = '.'.join([network, station])
            coordinates = AttribDict()
            for key in ['latitude', 'longitude', 'elevation']:
                coordinates[key] = inv[netsta][key]
            for tr in stream:
                # add coordinates
                tr.stats['coordinates'] = coordinates
                # add PAZ
                entries = inv[tr.id]
                if len(entries) > 1:
                    # multiple entries found
                    for entry in entries:
                        # trim current trace to timespan of current entry
                        temp = tr.slice(entry.starttime,
                                        entry.get('endtime', None))
                        # append valid paz
                        if 'paz' not in entry:
                            raise ArcLinkException(MSG_NOPAZ)
                        temp.stats['paz'] = entry.paz
                        # add to end of stream
                        stream.append(temp)
                    # remove split trace
                    stream.remove(tr)
                else:
                    # single entry found - apply direct
                    entry = entries[0]
                    if 'paz' not in entry:
                        raise ArcLinkException(MSG_NOPAZ)
                    tr.stats['paz'] = entry.paz
        return stream

    def saveWaveform(self, filename, network, station, location, channel,
                     starttime, endtime, format="MSEED", compressed=True,
                     route=True, unpack=True):
        """
        Writes a retrieved waveform directly into a file.

        This method ensures the storage of the unmodified waveform data
        delivered by the ArcLink server, e.g. preserving the record based
        quality flags of MiniSEED files which would be neglected reading it
        with :mod:`obspy.mseed`.

        :type filename: str
        :param filename: Name of the output file.
        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :type location: str
        :param location: Location code, e.g. ``'01'``. Location code may
            contain wild cards.
        :type channel: str
        :param channel: Channel code, e.g. ``'EHE'``. Channel code may
            contain wild cards.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type format: str, optional
        :param format: Output format. Either as full SEED (``'FSEED'``) or
            Mini-SEED (``'MSEED'``) volume. Defaults to ``'MSEED'``.

            .. note::
                A format ``'XSEED'`` is documented, but not yet implemented in
                ArcLink.
        :type compressed: bool, optional
        :param compressed: Request compressed files from ArcLink server.
            Default is ``True``.
        :type route: bool, optional
        :param route: Enables ArcLink routing. Default is ``True``.
        :type unpack: bool, optional
        :param unpack: Unpack compressed waveform files before storing to disk.
            Default is ``True``.
        :return: None

        .. rubric:: Example

        >>> from obspy.arclink import Client
        >>> from obspy import UTCDateTime
        >>> client = Client("webdc.eu", 18001, user='test@obspy.org')
        >>> t = UTCDateTime(2009, 1, 1, 12, 0)
        >>> client.saveWaveform('BW.MANZ.fullseed', 'BW', 'MANZ', '', '*',
        ...                     t, t + 20, format='FSEED')  # doctest: +SKIP
        """
        format = format.upper()
        if format not in ["MSEED", "FSEED"]:
            msg = ("'%s' is not a valid format. Choose either 'MSEED' or "
                   "'FSEED'")
            raise ArcLinkException(msg)
        # check parameters
        is_name = isinstance(filename, (str, native_str))
        if not is_name and not hasattr(filename, "write"):
            msg = "Parameter filename must be either string or file handler."
            raise TypeError(msg)
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
        # check if data is still encrypted
        if data.startswith(b'Salted__'):
            # set "good" file names
            if is_name:
                if compressed and not filename.endswith('.bz2.openssl'):
                    filename += '.bz2.openssl'
                elif not compressed and not filename.endswith('.openssl'):
                    filename += '.openssl'
            # warn user that encoded files will not be unpacked
            if unpack:
                warnings.warn("Cannot unpack encrypted waveforms.")
        else:
            # not encoded - handle as usual
            if compressed:
                # unpack compressed data if option unpack is set
                if unpack:
                    data = bz2.decompress(data)
                elif is_name and not filename.endswith('.bz2'):
                    # set "good" file names
                    filename += '.bz2'
        # create file handler if a file name is given
        if is_name:
            fh = open(filename, "wb")
        else:
            fh = filename
        fh.write(data)
        if is_name:
            fh.close()

    def getRouting(self, network, station, starttime, endtime,
                   modified_after=None):
        """
        Get primary ArcLink host for given network/stations/time combination.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type modified_after: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param modified_after: Returns only data modified after given date.
            Default is ``None``, returning all available data.
        :return: Dictionary of host names.
        """
        # request type
        rtype = 'REQUEST ROUTING '
        if modified_after:
            rtype += 'modified_after=%s ' % modified_after.formatArcLink()
        # request data
        rdata = [starttime, endtime, network, station]
        # fetch plain XML document
        result = self._fetch(rtype, rdata, route=False)
        # parse XML document
        xml_doc = etree.fromstring(result)
        # get routing version
        if _ROUTING_NS_1_0 in xml_doc.nsmap.values():
            xml_ns = _ROUTING_NS_1_0
        elif _ROUTING_NS_0_1 in xml_doc.nsmap.values():
            xml_ns = _ROUTING_NS_0_1
        else:
            msg = "Unknown routing namespace %s"
            raise ArcLinkException(msg % xml_doc.nsmap)
        # convert into dictionary
        result = {}
        for route in xml_doc.xpath('ns0:route', namespaces={'ns0': xml_ns}):
            if xml_ns == _ROUTING_NS_0_1:
                # no location/stream codes in 0.1
                id = route.get('net_code') + '.' + route.get('sta_code') + '..'
            else:
                id = route.get('networkCode') + '.' + \
                    route.get('stationCode') + '.' + \
                    route.get('locationCode') + '.' + \
                    route.get('streamCode')
            result[id] = []
            for node in route.xpath('ns0:arclink', namespaces={'ns0': xml_ns}):
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
                # address field may contain multiple addresses (?)
                address = node.get('address').split(',')[0]
                temp['host'] = address.split(':')[0].strip()
                temp['port'] = int(address.split(':')[1].strip())
                result[id].append(temp)
        return result

    def _findRoute(self, routes, request_data):
        """
        Searches routing table for requested stream id and date/times.
        """
        # Note: Filtering by date/times is not really supported by ArcLink yet,
        # therefore not included here
        # Multiple fitting entries are sorted by priority only
        net, sta, cha, loc = (request_data + ['', ''])[2:6]
        keys = []
        for key in routes:
            parts = key.split('.')
            if parts[0] and net != '*' and not fnmatch(parts[0], net):
                continue
            if parts[1] and sta != '*' and not fnmatch(parts[1], sta):
                continue
            if parts[2] and loc != '*' and not fnmatch(parts[2], loc):
                continue
            if parts[3] and cha != '*' and not fnmatch(parts[3], cha):
                continue
            keys.append(key)
        if not keys:
            # no route found
            return False
        # merge all
        out = []
        for key in keys:
            temp = routes[key]
            if temp == []:
                out.append({})
            else:
                out.extend(temp)
        # sort by priority
        out = sorted(out, key=lambda x: x.get('priority', 1000))
        return out

    def getQC(self, network, station, location, channel, starttime,
              endtime, parameters='*', outages=True, logs=True):
        """
        Retrieve QC information of ArcLink streams.

        .. note::
            Requesting QC is documented but seems not to work at the moment.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :type location: str
        :param location: Location code, e.g. ``'01'``.
        :type channel: str
        :param channel: Channel code, e.g. ``'EHE'``.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type parameters: str, optional
        :param parameters: Comma-separated list of QC parameters. The following
            QC parameters are implemented in the present version:
            ``'availability'``, ``'delay'``, ``'gaps count'``,
            ``'gaps interval'``, ``'gaps length'``, ``'latency'``,
            ``'offset'``, ``'overlaps count'``, ``'overlaps interval'``,
            ``'overlaps length'``, ``'rms'``, ``'spikes amplitude'``,
            ``'spikes count'``, ``'spikes interval'``, ``'timing quality'``
            (default is ``'*'`` for requesting all parameters).
        :type outages: bool, optional
        :param outages: Include list of outages (default is ``True``).
        :type logs: bool, optional
        :param logs: Include log messages (default is ``True``).
        :return: XML document as string.
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

    @deprecated_keywords({'getPAZ': None, 'getCoordinates': None})
    def getMetadata(self, network, station, location, channel, starttime=None,
                    endtime=None, time=None, route=True):
        """
        Returns poles, zeros, normalization factor and sensitivity and station
        coordinates for a single channel at a given time.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :type location: str
        :param location: Location code, e.g. ``'01'``.
        :type channel: str
        :param channel: Channel code, e.g. ``'EHE'``.
        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Date and time.
        :type route: bool, optional
        :param route: Enables ArcLink routing (default is ``True``).
        :return: Dictionary containing keys 'paz' and 'coordinates'.


        .. rubric:: Example

        >>> from obspy.arclink import Client
        >>> from obspy import UTCDateTime
        >>> client = Client("webdc.eu", 18001, user='test@obspy.org')
        >>> t = UTCDateTime(2009, 1, 1)
        >>> data = client.getMetadata('BW', 'MANZ', '', 'EHZ', t)
        >>> data  # doctest: +NORMALIZE_WHITESPACE +SKIP
        {'paz': AttribDict({'poles': [(-0.037004+0.037016j),
                                      (-0.037004-0.037016j), (-251.33+0j),
                                      (-131.04-467.29j), (-131.04+467.29j)],
                            'sensitivity': 2516778600.0, 'zeros': [0j, 0j],
                            'name': 'LMU:STS-2/N/g=1500',
                            'normalization_factor': 60077000.0}),
        'coordinates': AttribDict({'latitude': 49.9862, 'elevation': 635.0,
                                   'longitude': 12.1083})}
        """
        # XXX: deprecation handling
        if starttime and endtime:
            # warn if old scheme
            msg = "The 'starttime' and 'endtime' keywords will be " + \
                "deprecated. Please use 'time' instead."
            warnings.warn(msg, category=DeprecationWarning)
        elif starttime and not endtime:
            # use a single starttime as time keyword
            time = starttime
            endtime = time + 0.00001
        elif not time:
            # if not temporal keyword is given raise an exception
            raise ValueError("keyword 'time' is required")
        else:
            # time is given
            starttime = time
            endtime = time + 0.000001
        # check if single trace
        id = '.'.join([network, station, location, channel])
        if '*' in id:
            msg = 'getMetadata supports only a single channel, use ' + \
                  'getInventory instead'
            raise ArcLinkException(msg)
        # fetch inventory
        result = self.getInventory(network=network, station=station,
                                   location=location, channel=channel,
                                   starttime=starttime, endtime=endtime,
                                   instruments=True, route=route)
        data = {}
        # paz
        id = '.'.join([network, station, location, channel])
        # HACK: returning first PAZ only for now - should happen only for a
        # timespan and not a single time
        if len(result[id]) > 1:
            msg = "Multiple PAZ found for %s. Applying first PAZ."
            warnings.warn(msg % (id), UserWarning)
        data['paz'] = result[id][0].paz
        # coordinates
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

        # Response type: A=Laplace(rad/s), B=Analog(Hz), C, D
        try:
            paz['response_type'] = xml_doc.get('type')
        except:
            paz['response_type'] = None

        # normalization factor
        try:
            if xml_ns == _INVENTORY_NS_1_0:
                paz['normalization_factor'] = \
                    float(xml_doc.get('normalizationFactor'))
            else:
                paz['normalization_factor'] = float(xml_doc.get('norm_fac'))
        except:
            paz['normalization_factor'] = None

        try:
            if xml_ns == _INVENTORY_NS_1_0:
                paz['normalization_frequency'] = \
                    float(xml_doc.get('normalizationFrequency'))
            else:
                paz['normalization_frequency'] = \
                    float(xml_doc.get('norm_freq'))
        except:
            paz['normalization_frequency'] = None

        # for backwards compatibility (but this is wrong naming!)
        paz['gain'] = paz['normalization_factor']

        # zeros
        paz['zeros'] = []
        if xml_ns == _INVENTORY_NS_1_0:
            nzeros = int(xml_doc.get('numberOfZeros', 0))
        else:
            nzeros = int(xml_doc.get('nzeros', 0))
        try:
            zeros = xml_doc.xpath('ns:zeros/text()',
                                  namespaces={'ns': xml_ns})[0]
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
        if xml_ns == _INVENTORY_NS_1_0:
            npoles = int(xml_doc.get('numberOfPoles', 0))
        else:
            npoles = int(xml_doc.get('npoles', 0))
        try:
            poles = xml_doc.xpath('ns:poles/text()',
                                  namespaces={'ns': xml_ns})[0]
            temp = poles.strip().replace(' ', '').replace(')(', ') (')
            for poles in temp.split():
                paz['poles'].append(complexifyString(poles))
        except:
            pass
        # check number of poles
        if len(paz['poles']) != npoles:
            raise ArcLinkException('Could not parse all poles')
        return paz

    def getPAZ(self, network, station, location, channel, starttime=None,
               endtime=None, time=None, route=False):
        """
        Returns poles, zeros, normalization factor and sensitivity for a
        single channel at a given time.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :type location: str
        :param location: Location code, e.g. ``'01'``.
        :type channel: str
        :param channel: Channel code, e.g. ``'EHE'``.
        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Date and time.
        :type route: bool, optional
        :param route: Enables ArcLink routing. Defaults to ``True``.
        :return: Dictionary containing PAZ information.

        .. rubric:: Example

        >>> from obspy.arclink import Client
        >>> from obspy import UTCDateTime
        >>> client = Client("webdc.eu", 18001, user='test@obspy.org')
        >>> t = UTCDateTime(2009, 1, 1)
        >>> paz = client.getPAZ('BW', 'MANZ', '', 'EHZ', t)
        >>> paz  # doctest: +NORMALIZE_WHITESPACE +SKIP
        AttribDict({'poles': [(-0.037004+0.037016j), (-0.037004-0.037016j),
                              (-251.33+0j), (-131.04-467.29j),
                              (-131.04+467.29j)],
                    'sensitivity': 2516778600.0,
                    'zeros': [0j, 0j],
                    'name': 'LMU:STS-2/N/g=1500',
                    'normalization_factor': 60077000.0})
        """
        # XXX: deprecation handling
        if starttime and endtime:
            # warn if old scheme
            msg = "The 'starttime' and 'endtime' keywords will be " + \
                "deprecated. Please use 'time' instead. Be aware that the" + \
                "result of getPAZ() will differ using the 'time' keyword."
            warnings.warn(msg, category=DeprecationWarning)
        elif starttime and not endtime:
            # use a single starttime as time keyword
            time = starttime
            endtime = time + 0.00001
        elif not time:
            # if not temporal keyword is given raise an exception
            raise ValueError("keyword 'time' is required")
        else:
            # time is given
            starttime = time
            endtime = time + 0.000001
        # check if single trace
        id = '.'.join([network, station, location, channel])
        if '*' in id:
            msg = 'getPAZ supports only a single channel, use getInventory' + \
                  ' instead'
            raise ArcLinkException(msg)
        # fetch inventory
        result = self.getInventory(network=network, station=station,
                                   location=location, channel=channel,
                                   starttime=starttime, endtime=endtime,
                                   instruments=True, route=route)
        try:
            if time is None:
                # old deprecated schema (ARGS!!!!)
                # HACK: returning first PAZ only for now
                if len(result[id]) > 1:
                    msg = "Multiple PAZ found for %s. Applying first PAZ."
                    warnings.warn(msg % (id), UserWarning)
                paz = result[id][0].paz
                return {paz.name: paz}
            else:
                # new schema
                return result[id][0].paz
        except:
            msg = 'Could not find PAZ for channel %s' % id
            raise ArcLinkException(msg)

    def saveResponse(self, filename, network, station, location, channel,
                     starttime, endtime, format='SEED'):
        """
        Writes response information into a file.

        :type filename: str or file
        :param filename: Name of the output file or open file like object.
        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``.
        :type location: str
        :param location: Location code, e.g. ``'01'``.
        :type channel: str
        :param channel: Channel code, e.g. ``'EHE'``.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type format: str, optional
        :param format: Output format. Currently only Dataless SEED (``'SEED'``)
            is supported.
        :return: None

        .. rubric:: Example

        >>> from obspy.arclink import Client
        >>> from obspy import UTCDateTime
        >>> client = Client("webdc.eu", 18001, user='test@obspy.org')
        >>> t = UTCDateTime(2009, 1, 1)
        >>> client.saveResponse('BW.MANZ..EHZ.dataless', 'BW', 'MANZ', '', '*',
        ...                     t, t + 1, format="SEED")  # doctest: +SKIP
        """
        # check format
        format = format.upper()

        if format == "SEED":
            # request type
            rtype = 'REQUEST RESPONSE format=%s' % format
            # request data
            rdata = [starttime, endtime, network, station, channel, location]
            # fetch dataless
            data = self._fetch(rtype, rdata)
        else:
            raise ValueError("Unsupported format %s" % format)
        if hasattr(filename, "write") and hasattr(filename.write, "__call__"):
            filename.write(data)
        else:
            with open(filename, "wb") as fp:
                fp.write(data)

    def getInventory(self, network, station='*', location='*', channel='*',
                     starttime=UTCDateTime(), endtime=UTCDateTime(),
                     instruments=False, route=True, sensortype='',
                     min_latitude=None, max_latitude=None,
                     min_longitude=None, max_longitude=None,
                     restricted=None, permanent=None, modified_after=None):
        """
        Returns information about the available networks and stations in that
        particular space/time region.

        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :type station: str
        :param station: Station code, e.g. ``'MANZ'``. Station code may contain
            wild cards.
        :type location: str
        :param location: Location code, e.g. ``'01'``. Location code may
            contain wild cards.
        :type channel: str
        :param channel: Channel code, e.g. ``'EHE'``. Channel code may contain
            wild cards.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type instruments: bool, optional
        :param instruments: Include instrument data (default is ``False``).
        :type route: bool, optional
        :param route: Enables ArcLink routing (default is ``True``).
        :type sensortype: str, optional
        :param sensortype: Limit streams to those using specific sensor types:
            ``"VBB"``, ``"BB"``, ``"SM"``, ``"OBS"``, etc. Can be also a
            combination like ``"VBB+BB+SM"``.
        :type min_latitude: float, optional
        :param min_latitude: Minimum latitude.
        :type max_latitude: float, optional
        :param max_latitude: Maximum latitude.
        :type min_longitude: float, optional
        :param min_longitude: Minimum longitude.
        :type max_longitude: float, optional
        :param max_longitude: Maximum longitude
        :type permanent: bool, optional
        :param permanent: Requesting only permanent or temporary networks
            respectively. Default is ``None``, therefore requesting all data.
        :type restricted: bool, optional
        :param restricted: Requesting only networks/stations/streams that have
            restricted or open data respectively. Default is ``None``.
        :type modified_after: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param modified_after: Returns only data modified after given date.
            Default is ``None``, returning all available data.
        :return: Dictionary of inventory information.

        .. rubric:: Example

        >>> from obspy.arclink import Client
        >>> client = Client("webdc.eu", 18001, user='test@obspy.org')
        >>> inv = client.getInventory('BW', 'M*', '*', 'EHZ', restricted=False,
        ...                           permanent=True, min_longitude=12,
        ...                           max_longitude=12.2) #doctest: +SKIP
        >>> inv.keys()  # doctest: +SKIP
        ['BW.MROB', 'BW.MANZ..EHZ', 'BW', 'BW.MANZ', 'BW.MROB..EHZ']
        >>> inv['BW']  # doctest: +SKIP
        AttribDict({'description': 'BayernNetz', 'region': 'Germany', ...
        >>> inv['BW.MROB']  # doctest: +SKIP
        AttribDict({'code': 'MROB', 'description': 'Rosenbuehl, Bavaria', ...
        """
        # request type
        rtype = 'REQUEST INVENTORY '
        if instruments:
            rtype += 'instruments=true '
        if modified_after:
            rtype += 'modified_after=%s ' % modified_after.formatArcLink()
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
        if _INVENTORY_NS_1_0 in xml_doc.nsmap.values():
            xml_ns = _INVENTORY_NS_1_0
            stream_ns = 'sensorLocation'
            component_ns = 'stream'
            seismometer_ns = 'sensor'
            name_ns = 'publicID'
            resp_paz_ns = 'responsePAZ'
        elif _INVENTORY_NS_0_2 in xml_doc.nsmap.values():
            xml_ns = _INVENTORY_NS_0_2
            stream_ns = 'seis_stream'
            component_ns = 'component'
            seismometer_ns = 'seismometer'
            name_ns = 'name'
            resp_paz_ns = 'resp_paz'
        else:
            msg = "Unknown inventory namespace %s"
            raise ArcLinkException(msg % xml_doc.nsmap)

        sensors = {}
        for sensor in xml_doc.xpath('ns:sensor', namespaces={'ns': xml_ns}):
            entry = {}
            for key in ['description', 'manufacturer', 'model', 'name', 'type',
                        'unit', 'response']:
                entry[key] = sensor.get(key, '')
            sensors[entry['response']] = entry

        # convert into dictionary
        data = AttribDict()
        for network in xml_doc.xpath('ns:network', namespaces={'ns': xml_ns}):
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
                net.remark = network.xpath(
                    'ns:remark', namespaces={'ns': xml_ns})[0].text or ''
            except:
                net.remark = ''
            # write network entries
            data[net.code] = net
            # stations
            for station in network.xpath('ns0:station',
                                         namespaces={'ns0': xml_ns}):
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
                    sta.remark = station.xpath(
                        'ns:remark', namespaces={'ns': xml_ns})[0].text or ''
                except:
                    sta.remark = ''
                # write station entry
                data[net.code + '.' + sta.code] = sta
                # instruments
                for stream in station.xpath('ns:' + stream_ns,
                                            namespaces={'ns': xml_ns}):
                    # fetch component
                    for comp in stream.xpath('ns:' + component_ns,
                                             namespaces={'ns': xml_ns}):
                        # date / times
                        try:
                            start = UTCDateTime(comp.get('start'))
                        except:
                            start = None
                        try:
                            end = UTCDateTime(comp.get('end'))
                        except:
                            end = None
                        # check date/time boundaries
                        if start > endtime:
                            continue
                        if end and starttime > end:
                            continue
                        if xml_ns == _INVENTORY_NS_0_2:
                            seismometer_id = stream.get(seismometer_ns, None)
                        else:
                            seismometer_id = comp.get(seismometer_ns, None)
                        # channel id
                        if xml_ns == _INVENTORY_NS_0_2:
                            # channel code is split into two attributes
                            id = '.'.join([net.code, sta.code,
                                           stream.get('loc_code', ''),
                                           stream.get('code', '  ') +
                                           comp.get('code', ' ')])
                        else:
                            id = '.'.join([net.code, sta.code,
                                           stream.get('code', ''),
                                           comp.get('code', '')])
                        # write channel entry
                        if id not in data:
                            data[id] = []
                        temp = AttribDict()
                        data[id].append(temp)

                        # fetch sensitivity etc
                        try:
                            temp['sensitivity'] = float(comp.get('gain'))
                        except:
                            temp['sensitivity'] = None
                        # again keep it backwards compatible
                        temp['gain'] = temp['sensitivity']
                        try:
                            temp['sensitivity_frequency'] = \
                                float(comp.get('gainFrequency'))
                        except:
                            temp['sensitivity_frequency'] = None
                        try:
                            temp['sensitivity_unit'] = comp.get('gainUnit')
                        except:
                            temp['sensitivity_unit'] = None

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
                        paz_id = xml_doc.xpath('ns:' + seismometer_ns +
                                               '[@' + name_ns + '="' +
                                               seismometer_id + '"]/@response',
                                               namespaces={'ns': xml_ns})
                        if not paz_id:
                            continue
                        paz_id = paz_id[0]
                        # hack for 0.2 schema
                        if paz_id.startswith('paz:'):
                            paz_id = paz_id[4:]
                        xml_paz = xml_doc.xpath('ns:' + resp_paz_ns + '[@' +
                                                name_ns + '="' + paz_id + '"]',
                                                namespaces={'ns': xml_ns})
                        if not xml_paz:
                            continue
                        # parse PAZ
                        paz = self.__parsePAZ(xml_paz[0], xml_ns)

                        # convert from Hz (Analog) to rad/s (Laplace)
                        if paz['response_type'] == "B":
                            x2pi = lambda x: (x * 2 * pi)
                            paz['poles'] = list(map(x2pi, paz['poles']))
                            paz['zeros'] = list(map(x2pi, paz['zeros']))
                            paz['normalization_factor'] = \
                                paz['normalization_factor'] * (2 * pi) ** \
                                (len(paz['poles']) - len(paz['zeros']))
                            paz['gain'] = paz['normalization_factor']
                            paz['response_type'] = "A"

                        # sensitivity
                        paz['sensitivity'] = temp['sensitivity']
                        paz['sensitivity_frequency'] = \
                            temp['sensitivity_frequency']
                        paz['sensitivity_unit'] = temp['sensitivity_unit']
                        temp['paz'] = paz

                        # add some seismometer-specific "nice to have" stuff
                        publicID = xml_paz[0].get('publicID')
                        try:
                            paz['sensor_manufacturer'] = \
                                sensors[publicID]['manufacturer']
                            paz['sensor_model'] = sensors[publicID]['model']
                        except:
                            paz['sensor_manufacturer'] = None
                            paz['sensor_model'] = None

        return data

    def getNetworks(self, starttime, endtime):
        """
        Returns a dictionary of available networks within the given time span.

        .. note::
            Currently the time span seems to be ignored by the ArcLink servers,
            therefore all possible networks are returned.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :return: Dictionary of network data.
        """
        return self.getInventory(network='*', starttime=starttime,
                                 endtime=endtime, route=False)

    def getStations(self, starttime, endtime, network):
        """
        Returns a dictionary of available stations in the given network(s).

        .. note::
            Currently the time span seems to be ignored by the ArcLink servers,
            therefore all possible stations are returned.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start date and time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End date and time.
        :type network: str
        :param network: Network code, e.g. ``'BW'``.
        :return: Dictionary of station data.
        """
        data = self.getInventory(network=network, starttime=starttime,
                                 endtime=endtime)
        stations = [value for key, value in data.items()
                    if key.startswith(network + '.')
                    and "code" in value]
        return stations


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
