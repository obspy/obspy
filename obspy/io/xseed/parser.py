# -*- coding: utf-8 -*-
"""
Main module containing XML-SEED parser.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future import standard_library
from future.utils import native_str

import copy
import datetime
import io
import math
import os
import warnings
import zipfile

with standard_library.hooks():
    import urllib.request  # @UnresolvedImport

from lxml.etree import parse as xmlparse
from lxml.etree import Element, SubElement, tostring
import numpy as np

from obspy import Stream, Trace, __version__
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.decorator import map_example_filename, deprecated
from . import DEFAULT_XSEED_VERSION, blockette
from .utils import IGNORE_ATTR, SEEDParserException, to_tag


CONTINUE_FROM_LAST_RECORD = b'*'
HEADERS = ['V', 'A', 'S']
# @see: http://www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf, p. 24-26
HEADER_INFO = {
    'V': {'name': 'Volume Index Control Header',
          'blockettes': [10, 11, 12]},
    'A': {'name': 'Abbreviation Dictionary Control Header',
          'blockettes': [30, 31, 32, 33, 34, 41, 43, 44, 45, 46, 47, 48]},
    'S': {'name': 'Station Control Header',
          'blockettes': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]}
}
RESP_BLOCKETTES = [53, 54, 55, 56, 57, 58, 60, 61, 62]

XSEED_VERSIONS = ['1.0', '1.1']

# Index fields of the abbreviation blockettes.
INDEX_FIELDS = {30: 'data_format_identifier_code',
                31: 'comment_code_key',
                32: 'source_lookup_code',
                33: 'abbreviation_lookup_code',
                34: 'unit_lookup_code',
                35: 'beam_lookup_code'}


class Parser(object):
    """
    The XML-SEED parser class parses dataless or full SEED volumes.

    .. seealso::

        The SEED file format description can be found at
        http://www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf .

        The XML-SEED format was proposed in [Tsuboi2004]_.
    """

    def __init__(self, data=None, debug=False, strict=False,
                 compact=False):
        """
        Initializes the SEED parser.

        :type data: str, bytes, io.BytesIO or file
        :param data: Filename, URL, XSEED/SEED string, file pointer or
            BytesIO.
        :type debug: bool
        :param debug: Enables a verbose debug log during parsing of SEED file.
        :type strict: bool
        :param strict: Parser will raise an exception if SEED files does not
            stay within the SEED specifications.
        :type compact: bool
        :param compact: SEED volume will contain compact data strings. Missing
            time strings will be filled with 00:00:00.0000 if this option is
            disabled.
        """
        self.record_length = 4096
        self.version = 2.4
        self.blockettes = {}
        self.debug = debug
        self.strict = strict
        self.compact = compact
        self._format = None
        # All parsed data is organized in volume, abbreviations and a list of
        # stations.
        self.volume = None
        self.abbreviations = None
        self.stations = []
        # if a file name is given, read it directly to the parser object
        if data:
            self.read(data)

    def __str__(self):
        """
        """
        try:
            if len(self.stations) == 0:
                return 'No data'
        except:
            return 'No data'
        ret_str = ""
        inv = self.get_inventory()
        ret_str += "Networks:\n"
        # Sort alphabetically.
        networks = sorted(inv["networks"], key=lambda x: x["network_code"])
        for network in networks:
            ret_str += "\t%s (%s)\n" % (
                network["network_code"], network["network_name"])
        stations = sorted(inv["stations"], key=lambda x: x["station_id"])
        ret_str += "Stations:\n"
        for station in stations:
            ret_str += "\t%s (%s)\n" % (
                station["station_id"], station["station_name"])
        channels = sorted(inv["channels"], key=lambda x: x["channel_id"])
        ret_str += "Channels:\n"
        for channel in channels:
            start_date = channel["start_date"].strftime("%Y-%m-%d") if \
                channel["start_date"] else ""
            end_date = channel["end_date"].strftime("%Y-%m-%d") if \
                channel["end_date"] else ""
            ret_str += (
                "\t%s | %.2f Hz | %s | %s - %s | Lat: %.1f, Lng: %.1f\n") % \
                (channel["channel_id"], channel["sampling_rate"],
                 channel["instrument"], start_date, end_date,
                 channel["latitude"], channel["longitude"])
        return ret_str.strip()

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    @map_example_filename("data")
    def read(self, data):
        """
        General parser method for XML-SEED and Dataless SEED files.

        :type data: str, bytes, io.BytesIO or file
        :param data: Filename, URL or XSEED/SEED string as file pointer or
            BytesIO.
        """
        if getattr(self, "_format", None):
            warnings.warn("Clearing parser before every subsequent read()")
            self.__init__()
        # try to transform everything into BytesIO object
        if isinstance(data, (str, native_str)):
            if "://" in data:
                # some URL
                data = urllib.request.urlopen(data).read()
                data = io.BytesIO(data)
            elif os.path.isfile(data):
                # looks like a file - read it
                with open(data, 'rb') as f:
                    data = f.read()
                data = io.BytesIO(data)
            else:
                try:
                    data = data.encode()
                except:
                    pass
                try:
                    data = io.BytesIO(data)
                except:
                    raise IOError("data is neither filename nor valid URL")
        # but could also be a big string with data
        elif isinstance(data, bytes):
            data = io.BytesIO(data)
        elif not hasattr(data, "read"):
            raise TypeError
        # check first byte of data BytesIO object
        first_byte = data.read(1)
        data.seek(0)
        if first_byte.isdigit():
            # SEED volumes starts with a number
            self._parse_SEED(data)
            self._format = 'SEED'
        elif first_byte == b'<':
            # XML files should always starts with an '<'
            self._parse_XSEED(data)
            self._format = 'XSEED'
        else:
            raise IOError("First byte of data must be in [0-9<]")

    @deprecated("'getXSEED' has been renamed to 'get_XSEED'. "
                "Use that instead.")
    def getXSEED(self, *args, **kwargs):
        return self.get_XSEED(*args, **kwargs)

    def get_XSEED(self, version=DEFAULT_XSEED_VERSION, split_stations=False):
        """
        Returns a XSEED representation of the current Parser object.

        :type version: float, optional
        :param version: XSEED version string (default is ``1.1``).
        :type split_stations: bool, optional
        :param split_stations: Splits stations containing multiple channels
            into multiple documents.
        :rtype: str or dict
        :return: Returns either a string or a dict of strings depending
            on the flag ``split_stations``.
        """
        if version not in XSEED_VERSIONS:
            raise SEEDParserException("Unknown XML-SEED version!")
        doc = Element("xseed", version=version)
        # Nothing to write if not all necessary data is available.
        if not self.volume or not self.abbreviations or \
                len(self.stations) == 0:
            msg = 'No data to be written available.'
            raise SEEDParserException(msg)
        # Check blockettes:
        if not self._check_blockettes():
            msg = 'Not all necessary blockettes are available.'
            raise SEEDParserException(msg)
        # Add blockettes 11 and 12 only for XSEED version 1.0.
        if version == '1.0':
            self._create_blockettes_11_and_12(blockette12=True)
        # Now start actually filling the XML tree.
        # Volume header:
        sub = SubElement(doc, to_tag('Volume Index Control Header'))
        for blkt in self.volume:
            sub.append(blkt.get_XML(xseed_version=version))
        # Delete blockettes 11 and 12 if necessary.
        if version == '1.0':
            self._delete_blockettes_11_and_12()
        # Abbreviations:
        sub = SubElement(
            doc, to_tag('Abbreviation Dictionary Control Header'))
        for blkt in self.abbreviations:
            sub.append(blkt.get_XML(xseed_version=version))
        if not split_stations:
            # Don't split stations
            for station in self.stations:
                sub = SubElement(doc, to_tag('Station Control Header'))
                for blkt in station:
                    sub.append(blkt.get_XML(xseed_version=version))
            if version == '1.0':
                # To pass the XSD schema test an empty time span control header
                # is added to the end of the file.
                SubElement(doc, to_tag('Timespan Control Header'))
                # Also no data is present in all supported SEED files.
                SubElement(doc, to_tag('Data Records'))
            # Return single XML String.
            return tostring(doc, pretty_print=True, xml_declaration=True,
                            encoding='UTF-8')
        else:
            # generate a dict of XML resources for each station
            result = {}
            for station in self.stations:
                cdoc = copy.copy(doc)
                sub = SubElement(cdoc, to_tag('Station Control Header'))
                for blkt in station:
                    sub.append(blkt.get_XML(xseed_version=version))
                if version == '1.0':
                    # To pass the XSD schema test an empty time span control
                    # header is added to the end of the file.
                    SubElement(doc, to_tag('Timespan Control Header'))
                    # Also no data is present in all supported SEED files.
                    SubElement(doc, to_tag('Data Records'))
                try:
                    id = station[0].end_effective_date.datetime
                except AttributeError:
                    id = ''
                result[id] = tostring(cdoc, pretty_print=True,
                                      xml_declaration=True, encoding='UTF-8')
            return result

    @deprecated("'writeXSEED' has been renamed to 'write_XSEED'. "
                "Use that instead.")
    def writeXSEED(self, *args, **kwargs):
        return self.write_XSEED(*args, **kwargs)

    def write_XSEED(self, filename, *args, **kwargs):
        """
        Writes a XML-SEED file with given name.
        """
        result = self.get_XSEED(*args, **kwargs)
        if isinstance(result, bytes):
            with open(filename, 'wb') as f:
                f.write(result)
            return
        elif isinstance(result, dict):
            for key, value in result.items():
                if isinstance(key, datetime.datetime):
                    # past meta data - append timestamp
                    fn = filename.split('.xml')[0]
                    fn = "%s.%s.xml" % (filename, UTCDateTime(key).timestamp)
                else:
                    # current meta data - leave original file name
                    fn = filename
                with open(fn, 'wb') as f:
                    f.write(value)
            return
        else:
            raise TypeError

    @deprecated("'get_SEED' has been renamed to 'get_SEED'. "
                "Use that instead.")
    def getSEED(self, *args, **kwargs):
        return self.get_SEED(*args, **kwargs)

    def get_SEED(self, compact=False):
        """
        Returns a SEED representation of the current Parser object.
        """
        self.compact = compact
        # Nothing to write if not all necessary data is available.
        if not self.volume or not self.abbreviations or not self.stations:
            msg = 'No data to be written available.'
            raise SEEDParserException(msg)
        # Check blockettes:
        if not self._check_blockettes():
            msg = 'Not all necessary blockettes are available.'
            raise SEEDParserException(msg)
        # String to be written to:
        seed_string = b''
        cur_count = 1
        volume, abbreviations, stations = self._create_blockettes_11_and_12()
        # Delete Blockette 11 again.
        self._delete_blockettes_11_and_12()

        # Finally write the actual SEED String.
        def fmt_seed(cnt, i):
            return ('%06i' % cnt).encode('ascii', 'strict') + i

        for _i in volume:
            seed_string += fmt_seed(cur_count, _i)
            cur_count += 1
        for _i in abbreviations:
            seed_string += fmt_seed(cur_count, _i)
            cur_count += 1
        # Remove name of the stations.
        stations = [_i[1:] for _i in stations]
        for _i in stations:
            for _j in _i:
                seed_string += fmt_seed(cur_count, _j)
                cur_count += 1
        return seed_string

    @deprecated("'writeSEED' has been renamed to 'write_SEED'. "
                "Use that instead.")
    def writeSEED(self, *args, **kwargs):
        return self.write_SEED(*args, **kwargs)

    def write_SEED(self, filename, *args, **kwargs):
        """
        Writes a dataless SEED file with given name.
        """
        fh = open(filename, 'wb')
        fh.write(self.get_SEED(*args, **kwargs))
        fh.close()

    @deprecated("'getRESP' has been renamed to 'get_RESP'. "
                "Use that instead.")
    def getRESP(self, *args, **kwargs):
        return self.get_RESP(*args, **kwargs)

    def get_RESP(self):
        """
        Returns a RESP representation of the current Parser object.

        It aims to produce the same RESP files as when running rdseed with
        the command: "rdseed -f seed.test -R".
        """
        # Check if there are any stations at all.
        if len(self.stations) == 0:
            raise Exception('No data to be written.')
        filename = None
        # Channel Response list.
        resp_list = []
        # Loop over all stations.
        for station in self.stations:
            resp = io.BytesIO(b'')
            blockettes = []
            # Read the current station information and store it.
            cur_station = station[0].station_call_letters.strip()
            cur_network = station[0].network_code.strip()
            # Loop over all blockettes in that station.
            for _i in range(1, len(station)):
                # Catch all blockette 52.
                if station[_i].id == 52:
                    cur_location = station[_i].location_identifier.strip()
                    cur_channel = station[_i].channel_identifier.strip()
                    # Take old list and send it to the RESP parser.
                    _pos = resp.tell()
                    resp.seek(0, os.SEEK_END)
                    _len = resp.tell()
                    resp.seek(_pos)
                    if _len != 0:
                        # Send the blockettes to the parser and append to list.
                        self._get_RESP_string(resp, blockettes, cur_station)
                        resp_list.append([filename, resp])
                    # Create the file name.
                    filename = 'RESP.%s.%s.%s.%s' \
                        % (cur_network, cur_station, cur_location, cur_channel)
                    # Create new BytesIO and list.
                    resp = io.BytesIO(b'')
                    blockettes = []
                    blockettes.append(station[_i])
                    # Write header and the first two lines to the string.
                    header = \
                        '#\t\t<< obspy, Version %s >>\n' % __version__ + \
                        '#\t\t\n' + \
                        '#\t\t======== CHANNEL RESPONSE DATA ========\n' + \
                        'B050F03     Station:     %s\n' % cur_station + \
                        'B050F16     Network:     %s\n' % cur_network
                    # Write to BytesIO.
                    resp.write(header.encode('ascii', 'strict'))
                    continue
                blockettes.append(station[_i])
            # It might happen that no blockette 52 is specified,
            if len(blockettes) != 0:
                # One last time for the last channel.
                self._get_RESP_string(resp, blockettes, cur_station)
                resp_list.append([filename, resp])
        # Combine multiple channels.
        new_resp_list = []
        available_channels = [_i[0] for _i in resp_list]
        channel_set = set(available_channels)
        for channel in channel_set:
            channel_list = [_i for _i in resp_list if _i[0] == channel]
            if len(channel_list) == 1:
                new_resp_list.append(channel_list[0])
            else:
                for _i in range(1, len(channel_list)):
                    channel_list[_i][1].seek(0, 0)
                    channel_list[0][1].write(channel_list[_i][1].read())
                new_resp_list.append(channel_list[0])
        return new_resp_list

    def _select(self, seed_id, datetime=None):
        """
        Selects all blockettes related to given SEED id and datetime.
        """
        old_format = self._format
        # parse blockettes if not SEED. Needed for XSEED to be initialized.
        # XXX: Should potentially be fixed at some point.
        if self._format != 'SEED':
            self.__init__(self.get_SEED())
        if old_format == "XSEED":
            self._format = "XSEED"
        # split id
        if '.' in seed_id:
            net, sta, loc, cha = seed_id.split('.')
        else:
            cha = seed_id
            net = sta = loc = None
        # create a copy of station list
        stations = list(self.stations)
        # filter blockettes list by given SEED id
        station_flag = False
        channel_flag = False
        blockettes = []
        for station in stations:
            for blk in station:
                if blk.id == 50:
                    station_flag = False
                    if net is not None and blk.network_code != net:
                        continue
                    if sta is not None and blk.station_call_letters != sta:
                        continue
                    station_flag = True
                    tmpb50 = blk
                elif blk.id == 52 and station_flag:
                    channel_flag = False
                    if loc is not None and blk.location_identifier != loc:
                        continue
                    if blk.channel_identifier != cha:
                        continue
                    if datetime is not None:
                        if blk.start_date > datetime:
                            continue
                        if blk.end_date and blk.end_date < datetime:
                            continue
                    channel_flag = True
                    blockettes.append(tmpb50)
                    blockettes.append(blk)
                elif channel_flag and station_flag:
                    blockettes.append(blk)
        # check number of selected channels (equals number of blockette 52)
        b50s = [b for b in blockettes if b.id == 50]
        b52s = [b for b in blockettes if b.id == 52]
        if len(b50s) == 0 or len(b52s) == 0:
            msg = 'No channel found with the given SEED id: %s'
            raise SEEDParserException(msg % (seed_id))
        elif len(b50s) > 1 or len(b52s) > 1:
            msg = 'More than one channel found with the given SEED id: %s'
            raise SEEDParserException(msg % (seed_id))
        return blockettes

    @deprecated("'getPAZ' has been renamed to 'get_PAZ'. "
                "Use that instead.")
    def getPAZ(self, *args, **kwargs):
        return self.get_PAZ(*args, **kwargs)

    def get_PAZ(self, seed_id, datetime=None):
        """
        Return PAZ.

        .. note:: Currently only the Laplace transform is supported, that
            is blockettes 43 and 53. A UserWarning will be raised for
            unsupported response blockettes, however all other values, such
            as overall sensitivity, normalization constant, etc. will be still
            returned if found.

        :type seed_id: str
        :param seed_id: SEED or channel id, e.g. ``"BW.RJOB..EHZ"`` or
            ``"EHE"``.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param datetime: Timestamp of requested PAZ values
        :return: Dictionary containing PAZ as well as the overall
            sensitivity, the gain in the dictionary is the A0 normalization
            constant
        """
        blockettes = self._select(seed_id, datetime)
        data = {}
        for blkt in blockettes:
            if blkt.id == 58:
                if blkt.stage_sequence_number == 0:
                    data['sensitivity'] = blkt.sensitivity_gain
                elif blkt.stage_sequence_number == 1:
                    data['seismometer_gain'] = blkt.sensitivity_gain
                elif blkt.stage_sequence_number == 2:
                    data['digitizer_gain'] = blkt.sensitivity_gain
            elif blkt.id == 53 or blkt.id == 60:
                if blkt.id == 60:
                    abbreviation = blkt.stages[0][1]
                    data['seismometer_gain'] = \
                        [blk.sensitivity_gain for blk in self.abbreviations
                         if hasattr(blk, 'response_lookup_key') and
                            blk.response_lookup_key == abbreviation][0]
                    abbreviation = blkt.stages[0][0]
                    resp = [blk for blk in self.abbreviations
                            if hasattr(blk, 'response_lookup_key') and
                            blk.response_lookup_key == abbreviation][0]
                    label = 'response_type'
                else:
                    resp = blkt
                    label = 'transfer_function_types'
                # Check if Laplace transform
                if getattr(resp, label) not in ["A", "B"]:
                    msg = 'Only the Laplace (rad/sec) or Analog (Hz) ' + \
                          'transform response types are supported. ' + \
                          'Skipping other response information.'
                    warnings.warn(msg, UserWarning)
                    continue
                # A0_normalization_factor
                data['gain'] = resp.A0_normalization_factor
                # Poles
                data['poles'] = []
                for i in range(resp.number_of_complex_poles):
                    try:
                        p = complex(resp.real_pole[i], resp.imaginary_pole[i])
                    except TypeError:
                        p = complex(resp.real_pole, resp.imaginary_pole)
                    data['poles'].append(p)
                # Zeros
                data['zeros'] = []
                for i in range(resp.number_of_complex_zeros):
                    try:
                        z = complex(resp.real_zero[i], resp.imaginary_zero[i])
                    except TypeError:
                        z = complex(resp.real_zero, resp.imaginary_zero)
                    data['zeros'].append(z)
                # force conversion from Hz to Laplace
                if getattr(resp, label) == "B":
                    def x2pi(x):
                        return x * 2 * np.pi

                    data['poles'] = list(map(x2pi, data['poles']))
                    data['zeros'] = list(map(x2pi, data['zeros']))
                    data['gain'] = resp.A0_normalization_factor * \
                        (2 * np.pi) ** \
                        (len(data['poles']) - len(data['zeros']))
        return data

    @deprecated("'getCoordinates' has been renamed to 'get_coordinates'. "
                "Use that instead.")
    def getCoordinates(self, *args, **kwargs):
        return self.get_coordinates(*args, **kwargs)

    def get_coordinates(self, seed_id, datetime=None):
        """
        Return Coordinates (from blockette 52)

        :type seed_id: str
        :param seed_id: SEED or channel id, e.g. ``"BW.RJOB..EHZ"`` or
            ``"EHE"``.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param datetime: Timestamp of requested PAZ values
        :return: Dictionary containing Coordinates (latitude, longitude,
            elevation)
        """
        blockettes = self._select(seed_id, datetime)
        data = {}
        for blkt in blockettes:
            if blkt.id == 52:
                data['latitude'] = blkt.latitude
                data['longitude'] = blkt.longitude
                data['elevation'] = blkt.elevation
                data['local_depth'] = blkt.local_depth
                break
        return data

    @deprecated("'writeRESP' has been renamed to 'write_RESP'. "
                "Use that instead.")
    def writeRESP(self, *args, **kwargs):
        return self.write_RESP(*args, **kwargs)

    def write_RESP(self, folder, zipped=False):
        """
        Writes for each channel a RESP file within a given folder.

        :param folder: Folder name.
        :param zipped: Compresses all files into a single ZIP archive named by
            the folder name extended with the extension '.zip'.
        """
        new_resp_list = self.get_RESP()
        # Check if channel information could be found.
        if len(new_resp_list) == 0:
            msg = ("No channel information could be found. The SEED file "
                   "needs to contain information about at least one channel.")
            raise Exception(msg)
        if not zipped:
            # Write single files.
            for response in new_resp_list:
                if folder:
                    file = open(os.path.join(folder, response[0]), 'wb')
                else:
                    file = open(response[0], 'wb')
                response[1].seek(0, 0)
                file.write(response[1].read())
                file.close()
        else:
            # Create a ZIP archive.
            zip_file = zipfile.ZipFile(folder + os.extsep + "zip", "w")
            for response in new_resp_list:
                response[1].seek(0, 0)
                zip_file.writestr(response[0], response[1].read())
            zip_file.close()

    def _parse_SEED(self, data):
        """
        Parses through a whole SEED volume.

        It will always parse the whole file and skip any time span data.

        :type data: file or io.BytesIO
        """
        # Jump to the beginning of the file.
        data.seek(0)
        # Retrieve some basic data like version and record length.
        temp = data.read(8)
        # Check whether it starts with record sequence number 1 and a volume
        # index control header.
        if temp != b'000001V ':
            raise SEEDParserException("Expecting 000001V ")
        # The first blockette has to be Blockette 10.
        temp = data.read(3)
        if temp not in [b'010', b'008', b'005']:
            raise SEEDParserException("Expecting blockette 010, 008 or 005")
        # Skip the next four bytes containing the length of the blockette.
        # data.seek(4, 1)
        data.read(4)
        # Set the version.
        self.version = float(data.read(4))
        # Get the record length.
        length = pow(2, int(data.read(2)))
        # Test record length.
        data.seek(length)
        temp = data.read(6)
        if temp != b'000002':
            msg = "Got an invalid logical record length %d" % length
            raise SEEDParserException(msg)
        self.record_length = length
        if self.debug:
            print("RECORD LENGTH: %d" % (self.record_length))
        # Set all temporary attributes.
        self.temp = {'volume': [], 'abbreviations': [], 'stations': []}
        # Jump back to beginning.
        data.seek(0)
        # Read the first record.
        record = data.read(self.record_length)
        merged_data = b''
        record_type = None
        # Loop through file and pass merged records to _parse_merged_data.
        while record:
            record_continuation = (record[7:8] == CONTINUE_FROM_LAST_RECORD)
            same_record_type = (record[6:7].decode() == record_type)
            if record_type == 'S' and record[8:11] != b'050':
                record_continuation = True
            if record_continuation and same_record_type:
                # continued record
                merged_data += record[8:]
            else:
                self._parse_merged_data(merged_data.strip(), record_type)
                # first or new type of record
                record_type = record[6:7].decode()
                merged_data = record[8:]
                if record_type not in HEADERS:
                    # only parse headers, no data
                    merged_data = ''
                    record_type = None
                    break
            if self.debug:
                if not record_continuation:
                    print("========")
                print((record[0:8]))
            record = data.read(self.record_length)
        # Use parse once again.
        self._parse_merged_data(merged_data.strip(), record_type)
        # Update the internal structure to finish parsing.
        self._update_internal_SEED_structure()

    @deprecated("'getInventory' has been renamed to 'get_inventory'. "
                "Use that instead.")
    def getInventory(self, *args, **kwargs):
        return self.get_inventory(*args, **kwargs)

    def get_inventory(self):
        """
        Function returning a dictionary about whats actually in the Parser
        object.
        """
        info = {"networks": [], "stations": [], "channels": []}
        current_network = None
        current_station = None
        for station in self.stations:
            for blkt in station:
                if blkt.id == 50:
                    current_network = blkt.network_code.strip()
                    network_id = blkt.network_identifier_code
                    if isinstance(network_id, (str, native_str)):
                        new_id = ""
                        for _i in network_id:
                            if _i.isdigit():
                                new_id += _i
                        network_id = int(new_id)
                    network_name = self._get_abbreviation(network_id)
                    cur_nw = {"network_code": current_network,
                              "network_name": network_name}
                    if cur_nw not in info["networks"]:
                        info["networks"].append(cur_nw)
                    current_station = blkt.station_call_letters.strip()
                    cur_stat = {"station_id": "%s.%s" % (current_network,
                                                         current_station),
                                "station_name": blkt.site_name}
                    if cur_stat not in info["stations"]:
                        info["stations"].append(cur_stat)
                    continue
                if blkt.id == 52:
                    if current_network is None or current_station is None:
                        raise Exception("Something went wrong")
                    chan_info = {}
                    channel = blkt.channel_identifier.strip()
                    location = blkt.location_identifier.strip()
                    chan_info["channel_id"] = "%s.%s.%s.%s" % (
                        current_network, current_station, location, channel)
                    chan_info["sampling_rate"] = blkt.sample_rate
                    chan_info["instrument"] = \
                        self._get_abbreviation(blkt.instrument_identifier)
                    chan_info["start_date"] = blkt.start_date
                    chan_info["end_date"] = blkt.end_date
                    chan_info["latitude"] = blkt.latitude
                    chan_info["longitude"] = blkt.longitude
                    chan_info["elevation_in_m"] = blkt.elevation
                    chan_info["local_depth_in_m"] = blkt.local_depth
                    info["channels"].append(chan_info)
                    continue
        return info

    def _get_abbreviation(self, identifier_code):
        """
        Helper function returning the abbreviation for the given identifier
        code.
        """
        for blkt in self.abbreviations:
            if blkt.id != 33:
                continue
            if blkt.abbreviation_lookup_code != identifier_code:
                continue
            return blkt.abbreviation_description
        return ""

    def _parse_XSEED(self, data):
        """
        Parse a XML-SEED string.

        :type data: file or io.BytesIO
        """
        data.seek(0)
        root = xmlparse(data).getroot()
        xseed_version = root.get('version')
        headers = root.getchildren()
        # Set all temporary attributes.
        self.temp = {'volume': [], 'abbreviations': [], 'stations': []}
        # Parse volume which is assumed to be the first header. Only parse
        # blockette 10 and discard the rest.
        self.temp['volume'].append(
            self._parse_XML_blockette(headers[0].getchildren()[0], 'V',
                                      xseed_version))
        # Append all abbreviations.
        for blkt in headers[1].getchildren():
            self.temp['abbreviations'].append(
                self._parse_XML_blockette(blkt, 'A', xseed_version))
        # Append all stations.
        for control_header in headers[2:]:
            if not control_header.tag == 'station_control_header':
                continue
            self.temp['stations'].append([])
            for blkt in control_header.getchildren():
                self.temp['stations'][-1].append(
                    self._parse_XML_blockette(blkt, 'S', xseed_version))
        # Update internal values.
        self._update_internal_SEED_structure()

    def _get_RESP_string(self, resp, blockettes, station):
        """
        Takes a file like object and a list of blockettes containing all
        blockettes for one channel and writes them RESP like to the BytesIO.
        """
        blkt52 = blockettes[0]
        # The first blockette in the list always has to be Blockette 52.
        channel_info = {'Location': blkt52.location_identifier,
                        'Channel': blkt52.channel_identifier,
                        'Start date': blkt52.start_date,
                        'End date': blkt52.end_date}
        # Set location and end date default values or convert end time..
        if len(channel_info['Location']) == 0:
            channel_info['Location'] = '??'
        if not channel_info['End date']:
            channel_info['End date'] = 'No Ending Time'
        else:
            channel_info['End date'] = channel_info['End date'].format_seed()
        # Convert starttime.
        channel_info['Start date'] = channel_info['Start date'].format_seed()
        # Write Blockette 52 stuff.
        resp.write((
            'B052F03     Location:    %s\n' % channel_info['Location'] +
            'B052F04     Channel:     %s\n' % channel_info['Channel'] +
            'B052F22     Start date:  %s\n' % channel_info['Start date'] +
            'B052F23     End date:    %s\n' % channel_info['End date'] +
            '#\t\t=======================================\n'
            ).encode('ascii', 'strict'))
        # Write all other blockettes. Sort by stage number (0 at the end) and
        # the specified blockette id order.
        order = [53, 54, 55, 56, 60, 61, 62, 57, 58, 59]
        blockettes = sorted(
            blockettes[1:],
            key=lambda x: (x.stage_sequence_number
                           if (hasattr(x, "stage_sequence_number") and
                               x.stage_sequence_number)
                           else float("inf"), order.index(x.id)))

        for blkt in blockettes:
            if blkt.id not in RESP_BLOCKETTES:
                continue
            try:
                resp.write(blkt.get_RESP(
                    station, channel_info['Channel'], self.abbreviations))
            except AttributeError:
                msg = 'RESP output for blockette %s not implemented yet.'
                raise AttributeError(msg % blkt.id)

    def _parse_XML_blockette(self, XML_blockette, record_type, xseed_version):
        """
        Takes the lxml tree of any blockette and returns a blockette object.
        """
        # Get blockette number.
        blockette_id = int(XML_blockette.values()[0])
        if blockette_id in HEADER_INFO[record_type].get('blockettes', []):
            class_name = 'Blockette%03d' % blockette_id
            if not hasattr(blockette, class_name):
                raise SEEDParserException(
                    'Blockette %d not implemented!' % blockette_id)
            blockette_class = getattr(blockette, class_name)
            blockette_obj = blockette_class(debug=self.debug,
                                            strict=self.strict,
                                            compact=self.compact,
                                            version=self.version,
                                            record_type=record_type,
                                            xseed_version=xseed_version)
            blockette_obj.parse_XML(XML_blockette)
            return blockette_obj
        elif blockette_id != 0:
            msg = "Unknown blockette type %d found" % blockette_id
            raise SEEDParserException(msg)

    def _create_cut_and_flush_record(self, blockettes, record_type):
        """
        Takes all blockettes of a record and return a list of finished records.

        If necessary it will cut the record and return two or more flushed
        records.

        The returned records also include the control header type code and the
        record continuation code. Therefore the returned record will have the
        length self.record_length - 6. Other methods are responsible for
        writing the sequence number.

        It will always return a list with records.
        """
        length = self.record_length - 8
        return_records = []
        # Loop over all blockettes.
        record = b''
        for blockette_ in blockettes:
            blockette_.compact = self.compact
            rec_len = len(record)
            # Never split a blockette’s “length/blockette type” section across
            # records.
            if rec_len + 7 > length:
                # Flush the rest of the record if necessary.
                record += b' ' * (length - rec_len)
                return_records.append(record)
                record = b''
                rec_len = 0
            blockette_str = blockette_.get_SEED()
            # Calculate how much of the blockette is too long.
            overhead = rec_len + len(blockette_str) - length
            # If negative overhead: Write blockette.
            if overhead <= 0:
                record += blockette_str
            # Otherwise finish the record and start one or more new ones.
            else:
                record += blockette_str[:len(blockette_str) - overhead]
                # The record so far not written.
                rest_of_the_record = blockette_str[(len(blockette_str) -
                                                    overhead):]
                # Loop over the number of records to be written.
                for _i in range(
                        int(math.ceil(len(rest_of_the_record) /
                                      float(length)))):
                    return_records.append(record)
                    record = b''
                    # It doesn't hurt to index a string more than its length.
                    record = record + \
                        rest_of_the_record[_i * length: (_i + 1) * length]
        if len(record) > 0:
            return_records.append(record)
        # Flush last record
        return_records[-1] = return_records[-1] + b' ' * \
            (length - len(return_records[-1]))
        # Add control header and continuation code.
        b_record_type = record_type.encode('ascii', 'ignore')
        return_records[0] = b_record_type + b' ' + return_records[0]
        for _i in range(len(return_records) - 1):
            return_records[_i + 1] = b_record_type + b'*' + \
                return_records[_i + 1]
        return return_records

    def _check_blockettes(self):
        """
        Checks if all blockettes necessary for creating a SEED String are
        available.
        """
        if 10 not in [_i.id for _i in self.volume]:
            return False
        abb_blockettes = [_i.id for _i in self.abbreviations]
        if 30 not in abb_blockettes and 33 not in abb_blockettes and \
           34 not in abb_blockettes:
            return False
        # Check every station:
        for _i in self.stations:
            stat_blockettes = [_j.id for _j in _i]
            if 50 not in stat_blockettes and 52 not in stat_blockettes and \
               58 not in stat_blockettes:
                return False
        return True

    def _compare_blockettes(self, blkt1, blkt2):
        """
        Compares two blockettes.
        """
        for key in blkt1.__dict__.keys():
            # Continue if just some meta data.
            if key in IGNORE_ATTR:
                continue
            if blkt1.__dict__[key] != blkt2.__dict__[key]:
                return False
        return True

    def _update_internal_SEED_structure(self):
        """
        Takes everything in the self.temp dictionary and writes it into the
        volume, abbreviations and stations attributes of the class.

        The self.temp dictionary can only contain one seed volume with a
        correct structure.

        This method will try to merge everything, discard double entries and
        adjust abbreviations.

        It will also discard unnecessary blockettes that will be created
        again when writing SEED or XSEED.
        """
        # If called without a filled temporary dictionary do nothing.
        if not self.temp:
            return
        # Check if everything is empty.
        if not self.volume and not self.abbreviations and \
                len(self.stations) == 0:
            # Delete Blockette 11 and 12.
            self.volume = [i for i in self.temp['volume']
                           if i.id not in [11, 12]]
            self.abbreviations = self.temp['abbreviations']
            self.stations.extend(self.temp['stations'])
            del self.temp
        else:
            msg = 'Merging is an experimental feature and still contains ' + \
                  'a lot of errors!'
            warnings.warn(msg, UserWarning)
            # XXX: Sanity check for multiple Blockettes. Remove duplicates.
            # self._removeDuplicateAbbreviations()
            # Check the abbreviations.
            for blkt in self.temp['abbreviations']:
                id = blkt.blockette_type
                # Loop over all existing abbreviations and find those with the
                # same id and content.
                cur_index = 1
                # Helper variable.
                for ex_blkt in self.abbreviations:
                    if id != ex_blkt.blockette_type:
                        continue
                    # Raise the current index if it is the same blockette.
                    cur_index += 1
                    if not self._compare_blockettes(blkt, ex_blkt):
                        continue
                    # Update the current blockette and all abbreviations.
                    self._update_temporary_stations(
                        id, getattr(ex_blkt, INDEX_FIELDS[id]))
                    break
                else:
                    self._update_temporary_stations(id, cur_index)
                    # Append abbreviation.
                    setattr(blkt, INDEX_FIELDS[id], cur_index)
                    self.abbreviations.append(blkt)
            # Update the stations.
            self.stations.extend(self.temp['stations'])
            # XXX Update volume control header!

        # Also make the version of the format 2.4.
        self.volume[0].version_of_format = 2.4

    def _update_temporary_stations(self, blkt_id, index_nr):
        """
        Loops over all stations, finds the corresponding blockettes and changes
        all abbreviation lookup codes.
        """
        # Blockette dictionary which maps abbreviation IDs and and fields.
        index = {
            # Abbreviation Blockette : {Station Blockette: (Fields)}
            30: {52: (16,)},
            31: {51: (5,), 59: (5,)},
            33: {50: (10,), 52: (6,)},
            34: {52: (8, 9), 53: (5, 6), 54: (5, 6), 55: (4, 5)}
        }
        blockettes = index[blkt_id]
        # Loop over all stations.
        stations = self.temp['stations']
        for station in stations:
            for blkt in station:
                try:
                    fields = blockettes[blkt.blockette_type]
                except:
                    continue
                for field in fields:
                    setattr(blkt, blkt.get_fields()[field - 2].field_name,
                            index_nr)

    def _parse_merged_data(self, data, record_type):
        """
        This method takes any merged SEED record and writes its blockettes
        in the corresponding dictionary entry of self.temp.
        """
        if not data:
            return
        # Create BytesIO for easier access.
        data = io.BytesIO(data)
        # Do not do anything if no data is passed or if a time series header
        # is passed.
        if record_type not in HEADERS:
            return
        # Set standard values.
        blockette_length = 0
        blockette_id = -1
        # Find out what kind of record is being parsed.
        if record_type == 'S':
            # Create new station blockettes list.
            self.temp['stations'].append([])
            root_attribute = self.temp['stations'][-1]
        elif record_type == 'V':
            # Just one Volume header per file allowed.
            if len(self.temp['volume']):
                msg = 'More than one Volume index control header found!'
                raise SEEDParserException(msg)
            root_attribute = self.temp['volume']
        else:
            # Just one abbreviations header allowed!
            if len(self.temp['abbreviations']):
                msg = 'More than one Abbreviation Dictionary Control ' + \
                      'Headers found!'
                warnings.warn(msg, UserWarning)
            root_attribute = self.temp['abbreviations']
        # Loop over all blockettes in data.
        while blockette_id != 0:
            # remove spaces between blockettes
            while data.read(1) == b' ':
                continue
            data.seek(data.tell() - 1)
            try:
                blockette_id = int(data.read(3))
                blockette_length = int(data.read(4))
            except:
                break
            data.seek(data.tell() - 7)
            if blockette_id in HEADER_INFO[record_type].get('blockettes', []):
                class_name = 'Blockette%03d' % blockette_id
                if not hasattr(blockette, class_name):
                    raise SEEDParserException('Blockette %d not implemented!' %
                                              blockette_id)
                blockette_class = getattr(blockette, class_name)
                blockette_obj = blockette_class(debug=self.debug,
                                                strict=self.strict,
                                                compact=self.compact,
                                                version=self.version,
                                                record_type=record_type)
                blockette_obj.parse_SEED(data, blockette_length)
                root_attribute.append(blockette_obj)
                self.blockettes.setdefault(blockette_id,
                                           []).append(blockette_obj)
            elif blockette_id != 0:
                msg = "Unknown blockette type %d found" % blockette_id
                raise SEEDParserException(msg)
        # check if everything is parsed
        _pos = data.tell()
        data.seek(0, os.SEEK_END)
        _len = data.tell()
        data.seek(_pos)
        if _pos != _len:
            warnings.warn("There exist unparsed elements!")

    def _create_blockettes_11_and_12(self, blockette12=False):
        """
        Creates blockettes 11 and 12 for SEED writing and XSEED version 1.1
        writing.
        """
        # All the following unfortunately is necessary to get a correct
        # Blockette 11:
        # Start with the station strings to be able to write Blockette 11
        # later on. The created list will contain lists with the first item
        # being the corresponding station identifier code and each part of the
        # record being a separate item.
        stations = []
        # Loop over all stations.
        for _i in self.stations:
            station = []
            # Blockette 50 always should be the first blockette
            station.append(_i[0].station_call_letters)
            # Loop over blockettes.
            station.extend(self._create_cut_and_flush_record(_i, 'S'))
            stations.append(station)
        # Make abbreviations.
        abbreviations = self._create_cut_and_flush_record(self.abbreviations,
                                                          'A')
        abbr_length = len(abbreviations)
        cur_count = 1 + abbr_length
        while True:
            blkt11 = blockette.Blockette011()
            blkt11.number_of_stations = len(self.stations)
            stations_lengths = [cur_count + 1]
            for _i in [len(_i) - 1 for _i in stations][:-1]:
                stations_lengths.append(stations_lengths[-1] + _i)
            blkt11.sequence_number_of_station_header = stations_lengths
            blkt11.station_identifier_code = \
                [_i[0].station_call_letters for _i in self.stations]
            self.volume.append(blkt11)
            if blockette12:
                # Blockette 12 is also needed.
                blkt12 = blockette.Blockette012()
                blkt12.number_of_spans_in_table = 0
                self.volume.append(blkt12)
            volume = self._create_cut_and_flush_record(self.volume, 'V')
            if cur_count - abbr_length < len(volume):
                cur_count += len(volume) - 1
                self._delete_blockettes_11_and_12()
                continue
            break
        return volume, abbreviations, stations

    def _delete_blockettes_11_and_12(self):
        """
        Deletes blockette 11 and 12.
        """
        self.volume = [i for i in self.volume if i.id not in [11, 12]]

    def rotate_to_ZNE(self, stream):
        """
        Rotates the three components of a Stream to ZNE.

        Currently limited to rotating exactly three components covering exactly
        the same time span. The components can have arbitrary orientation and
        need not be orthogonal to each other. The output will be a new Stream
        object containing vertical, north, and east channels.

        :param stream: The stream object to rotate. Needs to have exactly three
            components, all the same length and timespan. Furthermore all
            components need to be described in the Parser object.
        """
        from obspy.signal.rotate import rotate2ZNE

        if len(stream) != 3:
            msg = "Stream needs to have three components."
            raise ValueError(msg)
        # Network, station and location need to be identical for all three.
        is_unique = len(set([(i.stats.starttime.timestamp,
                              i.stats.endtime.timestamp,
                              i.stats.npts,
                              i.stats.network,
                              i.stats.station,
                              i.stats.location) for i in stream])) == 1
        if not is_unique:
            msg = ("All the Traces need to cover the same time span and have "
                   "the same network, station, and location.")
            raise ValueError(msg)
        all_arguments = []

        for tr in stream:
            dip = None
            azimuth = None
            blockettes = self._select(tr.id, tr.stats.starttime)
            for blockette_ in blockettes:
                if blockette_.id != 52:
                    continue
                dip = blockette_.dip
                azimuth = blockette_.azimuth
                break
            if dip is None or azimuth is None:
                msg = "Dip and azimuth need to be available for every trace."
                raise ValueError(msg)
            all_arguments.extend([np.asarray(tr.data, dtype=np.float64),
                                  azimuth, dip])
        # Now rotate all three traces.
        z, n, e = rotate2ZNE(*all_arguments)

        # Assemble a new Stream object.
        common_header = {
            "network": stream[0].stats.network,
            "station": stream[0].stats.station,
            "location": stream[0].stats.location,
            "channel": stream[0].stats.channel[0:2],
            "starttime": stream[0].stats.starttime,
            "sampling_rate": stream[0].stats.sampling_rate}

        tr_z = Trace(data=z, header=common_header)
        tr_n = Trace(data=n, header=common_header)
        tr_e = Trace(data=e, header=common_header)

        # Fix the channel_codes
        tr_z.stats.channel += "Z"
        tr_n.stats.channel += "N"
        tr_e.stats.channel += "E"

        return Stream(traces=[tr_z, tr_n, tr_e])
