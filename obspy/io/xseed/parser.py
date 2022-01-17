# -*- coding: utf-8 -*-
"""
Main module containing XML-SEED, dataless SEED and RESP parser.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import collections
import copy
import datetime
import io
import itertools
import math
import os
import re
import warnings
import zipfile

from lxml import etree
from lxml.etree import parse as xmlparse
from lxml.etree import Element, SubElement, tostring
import numpy as np

from obspy import Stream, Trace, __version__
from obspy.core.inventory import (Response,
                                  PolesZerosResponseStage,
                                  CoefficientsTypeResponseStage,
                                  InstrumentSensitivity,
                                  ResponseStage,
                                  FIRResponseStage,
                                  ResponseListResponseStage,
                                  PolynomialResponseStage)
from obspy.core.inventory.response import ResponseListElement
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.base import download_to_file
from obspy.core.util.decorator import map_example_filename
from obspy.core.util.obspy_types import ComplexWithUncertainties
from . import DEFAULT_XSEED_VERSION, blockette, InvalidResponseError
from .blockette import (Blockette053, Blockette054, Blockette057, Blockette058,
                        Blockette061)
from .utils import (IGNORE_ATTR, SEEDParserException, to_tag)
from .fields import Loop, VariableString


CONTINUE_FROM_LAST_RECORD = b'*'
HEADERS = ['V', 'A', 'S']
# @see: https://www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf, p. 24-26
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
    Class parsing dataless and full SEED, X-SEED, and RESP files.

    .. seealso::

        The SEED file format description can be found at
        https://www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf .

        The XML-SEED format was proposed in [Tsuboi2004]_.

        The IRIS RESP format can be found at
        http://ds.iris.edu/ds/nodes/dmc/data/formats/resp/

    """
    def __init__(self, data=None, debug=False, strict=False, compact=False):
        """
        Initializes the SEED parser.

        :type data: str, bytes, io.BytesIO or file
        :param data: Filename, URL, XSEED/SEED/RESP string, file pointer or
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
        except Exception:
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
        General parser method for XML-SEED, Dataless SEED, and RESP files.

        :type data: str, bytes, io.BytesIO or file
        :param data: Filename, URL or XSEED/SEED/RESP string as file pointer or
            BytesIO.
        """
        # Import here to avoid circular imports.
        from .core import _is_resp

        if getattr(self, "_format", None):
            warnings.warn("Clearing parser before every subsequent read()")
            self.__init__()
        # try to transform everything into BytesIO object
        if isinstance(data, str):
            if re.search(r"://", data) is not None:
                url = data
                data = io.BytesIO()
                download_to_file(url=url, filename_or_buffer=data)
                data.seek(0, 0)
            elif os.path.isfile(data):
                if _is_resp(data):
                    # RESP filename
                    with open(data, 'r') as f:
                        data = f.read()
                    self._parse_resp(data)
                    return
                else:
                    with open(data, 'rb') as f:
                        data = f.read()
                    data = io.BytesIO(data)
            elif data.startswith('#'):
                # RESP data
                self._parse_resp(data)
                return
            else:
                try:
                    data = data.encode()
                except Exception:
                    pass
                try:
                    data = io.BytesIO(data)
                except Exception:
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
            self._parse_seed(data)
            self._format = 'SEED'
        elif first_byte == b'<':
            # XML files should always starts with an '<'
            try:
                self._parse_xseed(data)
            except Exception as e:
                # if it looks like the XML is not XSEED, tell the user
                if not is_xseed(data):
                    msg = ("Encountered an error during parsing XSEED file "
                           "(%s: %s). Note that the XSEED parser can not "
                           "parse StationXML. Please contact developers if "
                           "your file really is XML-SEED.")
                    raise Exception(msg % (e.__class__.__name__, str(e)))
                raise
            self._format = 'XSEED'
        else:
            raise IOError("First byte of data must be in [0-9<]")

    def get_xseed(self, version=DEFAULT_XSEED_VERSION, split_stations=False):
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
            sub.append(blkt.get_xml(xseed_version=version))
        # Delete blockettes 11 and 12 if necessary.
        if version == '1.0':
            self._delete_blockettes_11_and_12()
        # Abbreviations:
        sub = SubElement(
            doc, to_tag('Abbreviation Dictionary Control Header'))
        for blkt in self.abbreviations:
            sub.append(blkt.get_xml(xseed_version=version))
        if not split_stations:
            # Don't split stations
            for station in self.stations:
                sub = SubElement(doc, to_tag('Station Control Header'))
                for blkt in station:
                    sub.append(blkt.get_xml(xseed_version=version))
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
                    sub.append(blkt.get_xml(xseed_version=version))
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

    def write_xseed(self, filename, *args, **kwargs):
        """
        Writes a XML-SEED file with given name.
        """
        result = self.get_xseed(*args, **kwargs)
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

    def get_seed(self, compact=False):
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

    def write_seed(self, filename, *args, **kwargs):
        """
        Writes a dataless SEED file with given name.
        """
        fh = open(filename, 'wb')
        fh.write(self.get_seed(*args, **kwargs))
        fh.close()

    def get_resp(self):
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
                        self._get_resp_string(resp, blockettes, cur_station)
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
                self._get_resp_string(resp, blockettes, cur_station)
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
            self.__init__(self.get_seed())
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

    def get_paz(self, seed_id, datetime=None):
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
                elif blkt.stage_sequence_number == 3:
                    if 'digitizer_gain' in data:
                        data['digitizer_gain'] *= blkt.sensitivity_gain
                    else:
                        data['digitizer_gain'] = blkt.sensitivity_gain
            elif blkt.id == 53 or blkt.id == 60:
                # If we get a blockette 53 or 60 we should add these
                if 'zeros' not in data:
                    data['zeros'] = []
                if 'poles' not in data:
                    data['poles'] = []
                if 'gain' not in data:
                    data['gain'] = 1.
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
                data['gain'] *= resp.A0_normalization_factor
                # Poles
                for i in range(resp.number_of_complex_poles):
                    try:
                        p = complex(resp.real_pole[i], resp.imaginary_pole[i])
                    except TypeError:
                        p = complex(resp.real_pole, resp.imaginary_pole)
                    # Do conversion to Laplace poles
                    if getattr(resp, label) == "B":
                        p *= 2. * np.pi
                        data['gain'] *= 2. * np.pi
                    data['poles'].append(p)
                # Zeros
                for i in range(resp.number_of_complex_zeros):
                    try:
                        z = complex(resp.real_zero[i], resp.imaginary_zero[i])
                    except TypeError:
                        z = complex(resp.real_zero, resp.imaginary_zero)
                    # Do conversion to Laplace zeros
                    if getattr(resp, label) == "B":
                        z *= 2. * np.pi
                        data['gain'] *= 1.0 / (2.0 * np.pi)
                    data['zeros'].append(z)
        return data

    def get_coordinates(self, seed_id, datetime=None):
        """
        Return coordinates (from blockette 52) of a channel.

        :type seed_id: str
        :param seed_id: SEED or channel id, e.g. ``"BW.RJOB..EHZ"`` or
            ``"EHE"``.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param datetime: Timestamp of requested PAZ values
        :return: Dictionary containing coordinates (latitude, longitude,
            elevation, local_depth, dip, azimuth)
        """
        blockettes = self._select(seed_id, datetime)
        data = {}
        for blkt in blockettes:
            if blkt.id == 52:
                data['latitude'] = blkt.latitude
                data['longitude'] = blkt.longitude
                data['elevation'] = blkt.elevation
                data['local_depth'] = blkt.local_depth
                data['dip'] = blkt.dip
                data['azimuth'] = blkt.azimuth
                break
        return data

    def write_resp(self, folder, zipped=False):
        """
        Writes for each channel a RESP file within a given folder.

        :param folder: Folder name.
        :param zipped: Compresses all files into a single ZIP archive named by
            the folder name extended with the extension '.zip'.
        """
        new_resp_list = self.get_resp()
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

    def _parse_resp(self, data):
        """
        Reads RESP files.

        Reads IRIS RESP formatted data as produced with
        'rdseed -f seed.test -R'.

        :type data: file or io.BytesIO
        """
        def record_type_from_blocketteid(bid):
            for voltype in HEADER_INFO:
                if bid in HEADER_INFO[voltype]['blockettes']:
                    return voltype

        # First parse the data into a list of Blockettes
        blockettelist = []
        # List of fields
        blockettefieldlist = []

        # Pre-compile as called a lot.
        pattern = re.compile(r"^B(\d+)F(\d+)(?:-(\d+))?(.*)")
        field_pattern = re.compile(r":\s*(\S*)")
        comment_pattern = re.compile(r"^#.*\+")

        last_blockette_id = None
        # No fields with number 0 exist.
        last_field_number = 0
        for line in data.splitlines():
            m = re.match(pattern, line)
            if m:
                # g contains the following:
                #
                # g[0]: Blockette number as a string with leading 0.
                # g[1]: Field number as a string
                # g[2]: End field number for two multi field lines,
                #       e.g. B053F15-18
                # g[3]: Everything afterwards.
                g = m.groups()
                blockette_number = g[0]
                field_number = int(g[1])
                # A new blockette is starting.
                # Either
                # * when the blockette number increases or
                # * (for blockette 58) when the field number suddenly decreases
                # to 3 again.
                # This catches a rare issue where blockette 58 is repeated
                # twice (gain for a channel + total sensitivity) and the
                # comments don't contain any `+` which would alternatively
                # trigger a new blockette to be created.
                if (blockette_number != last_blockette_id) or \
                        ((blockette_number == "058") and
                         (field_number == 3) and
                         (field_number < last_field_number)):
                    if len(blockettefieldlist) > 0:
                        blockettelist.append(blockettefieldlist)
                        blockettefieldlist = list()
                    last_blockette_id = blockette_number
                    last_field_number = 0
                # Single field lines.
                if not g[2]:
                    # Units of blkts 41 and 61 are normal.
                    if blockette_number not in ["041", "061"] \
                            or "units lookup" in g[3]:
                        value = re.search(field_pattern, g[3]).groups()[0]
                    # Blockette 61 has the FIR coefficients which are a bit
                    # different.
                    else:
                        value = g[3].strip().split()[-1]
                    blockettefieldlist.append((blockette_number, g[1], value))
                    last_field_number = field_number
                # Multiple field lines.
                else:
                    first_field = int(g[1])
                    last_field = int(g[2])
                    _fields = g[3].split()
                    values = _fields[-(last_field - first_field + 1):]
                    for i, value in enumerate(values):
                        blockettefieldlist.append(
                            (blockette_number, first_field + i, value))
                    last_field_number = field_number
            elif re.match(comment_pattern, line):
                # Comment line with a + in it means blockette is
                # finished start a new one
                if len(blockettefieldlist) > 0:
                    blockettelist.append(blockettefieldlist)
                    blockettefieldlist = list()
                    last_blockette_id = blockette_number
                    last_field_number = 0
        # Add last blockette
        if len(blockettefieldlist) > 0:
            blockettelist.append(blockettefieldlist)

        # Now popule the parser object from the list.
        self.temp = {'volume': [], 'abbreviations': [], 'stations': []}
        # Make an empty blockette 10
        self.temp['volume'].append(blockette.Blockette010(
            debug=self.debug, strict=False, compact=False, record_type='V'))

        # Collect all required lookups here so they can later be written to
        # blockettes 34.
        unit_lookups = {}

        # Each loop will have all the fields for a single blockette.
        for blkt in blockettelist:
            # Split on new station.
            if blkt[0][0] == "050":
                self.temp["stations"].append([])

            class_name = 'Blockette%s' % blkt[0][0]
            blockette_class = getattr(blockette, class_name)
            record_type = record_type_from_blocketteid(blockette_class.id)
            blockette_obj = blockette_class(debug=self.debug,
                                            strict=False,
                                            compact=False,
                                            record_type=record_type)
            blockette_fields = (blockette_obj.default_fields +
                                blockette_obj.get_fields())
            unrolled_blockette_fields = []

            for bf in blockette_fields:
                if isinstance(bf, Loop):
                    for df in bf.data_fields:
                        unrolled_blockette_fields.append(df)
                else:
                    unrolled_blockette_fields.append(bf)
            blockette_fields = copy.deepcopy(unrolled_blockette_fields)

            # List of fields with fields used removed,
            # so unused can be set to default after
            unused_fields = blockette_fields[:]

            for (_, field_number, resp_value) in blkt:
                for bfield in blockette_fields:
                    if bfield.id != int(field_number):
                        continue
                    if isinstance(bfield, VariableString):
                        # Variable string needs terminator '~'
                        resp_value += '~'

                    # Units need to be put into the abbreviations. Luckily
                    # they all have "unit" in their field names.
                    if "unit" in bfield.field_name:
                        resp_value = resp_value.upper()
                        if resp_value not in unit_lookups:
                            unit_lookups[resp_value] = \
                                str(len(unit_lookups) + 1)
                        resp_value = unit_lookups[resp_value]
                    resp_data = io.BytesIO(resp_value.encode('utf-8'))

                    if (hasattr(bfield, 'length') and
                            bfield.length < len(resp_value)):
                        # RESP does not use the same length for floats
                        # as SEED does
                        bfield.length = len(resp_value)
                    bfield.parse_seed(blockette_obj, resp_data)
                    if bfield in unused_fields:
                        unused_fields.remove(bfield)
                    break

            default_field_names = [_i.field_name
                                   for _i in blockette_obj.default_fields]
            for bfield in unused_fields:
                # Only set the unused fields to default values that are part
                # of the default fields. Anything else is potentially wrong.
                if bfield.field_name not in default_field_names:
                    continue
                bfield.parse_seed(blockette_obj, None)

            # Collect all blockettes.
            self.temp["stations"][-1].append(blockette_obj)
            # Also collect all blockettes here.
            self.blockettes.setdefault(blockette_obj.id,
                                       []).append(blockette_obj)

        self.stations = self.temp["stations"]
        self.volume = self.temp["volume"]

        # We have to do another pass - Blockette 52 has the output unit of
        # the signal response which should be the same as the input unit of
        # the first filter stage. An awkward loop will follow!
        for station in self.stations:
            blkt52 = None
            for blkt in station:
                if blkt.id == 52:
                    blkt52 = blkt
                # Either take a 5X blockette with a stage sequence number of
                # one or take the first 4x blockette and assume its stages
                # sequence number is one.
                elif blkt52 and ((hasattr(blkt, "stage_sequence_number") and
                                  blkt.stage_sequence_number == 1) or
                                 blkt.id in list(range(41, 50))):
                    if hasattr(blkt, "stage_signal_input_units"):
                        key = "stage_signal_input_units"
                    elif hasattr(blkt, "signal_input_units"):
                        key = "signal_input_units"
                    else:
                        continue
                    blkt52.units_of_signal_response = getattr(blkt, key)
                    # Reset so each channel only gets the first.
                    blkt52 = None

        # One more to try to find the sampling rate of the last digitizer in
        # the chain.
        for station in self.stations:
            blkt52 = None
            for blkt in station:
                if blkt.id == 52:
                    blkt52 = blkt
                elif blkt.id in [47, 57]:
                    # Set it all the time - the last one will stick.
                    blkt52.sample_rate = \
                        int(round(blkt.input_sample_rate /
                                  blkt.decimation_factor))

        # Write all the abbreviations.
        mappings = {
            "COUNTS": "Digital Counts",
            "COUNTS/V": "Counts per Volt",
            "M": "Displacement in Meters",
            "M/S": "Velocity in Meters per Second",
            "M/S**2": "Acceleration in Meters Per Second Per Second",
            "M**3/M**3": "Volumetric Strain",
            "V": "Volts",
            "A": "Amperes"}
        for unit_name, lookup_key in unit_lookups.items():
            blkt = blockette.Blockette034()
            blkt.blockette_type = 34
            blkt.unit_lookup_code = int(lookup_key)
            blkt.unit_name = unit_name
            if unit_name in mappings:
                blkt.unit_description = mappings[unit_name]
            self.temp['abbreviations'].append(blkt)
            self.blockettes.setdefault(34, []).append(blkt)

        # Keep track of all used response lookup keys so they are not
        # duplicated.
        response_lookup_keys = set()

        # Reconstruct blockette 60 and move all the 4X blockettes to the
        # abbreviations. Another awkward and hard to understand loop.
        def _finalize_blkt60(stages):
            key = max(response_lookup_keys) if response_lookup_keys else 0
            stage_numbers = sorted(stages.keys())
            assert stage_numbers == list(range(min(stage_numbers),
                                               max(stage_numbers) + 1))
            stage_list = []
            for _i in stage_numbers:
                lookup_keys = []
                for s in stages[_i]:
                    key += 1
                    response_lookup_keys.add(key)
                    lookup_keys.append(key)
                    s.response_lookup_key = key
                    self.temp["abbreviations"].append(s)
                stage_list.append(lookup_keys)
            b = blockette.Blockette060()
            b.stages = stage_list
            return b

        new_stations = []
        for station in self.stations:
            blkts = []
            in_blkt_60 = False
            cur_blkt_stage = None
            cur_blkt_60_stages = collections.defaultdict(list)
            for b in station:
                if 40 <= b.id <= 49:
                    if not in_blkt_60:
                        msg = "4X blockette encountered outside of " \
                            "blockette 60"
                        raise ValueError(msg)
                    cur_blkt_60_stages[cur_blkt_stage].append(b)
                    continue
                # Otherwise.
                elif b.id == 60:
                    in_blkt_60 = True
                    # Always set in blockette 60 for RESP files.
                    cur_blkt_stage = b.stage_sequence_number
                    continue
                # Now assemble the new Blockette 60 if any.
                if in_blkt_60:
                    blkts.append(_finalize_blkt60(cur_blkt_60_stages))
                    in_blkt_60 = False
                    cur_blkt_stage = None
                    cur_blkt_60_stages = []
                # Just append all other blockettes.
                blkts.append(b)
            # Might still have one left.
            if in_blkt_60:
                blkts.append(_finalize_blkt60(cur_blkt_60_stages))
                in_blkt_60 = False
                cur_blkt_stage = None
                cur_blkt_60_stages = []
            new_stations.append(blkts)
        self.stations = new_stations

        self.abbreviations = self.temp["abbreviations"]

    def resolve_abbreviation(self, abbreviation_blockette_number, lookup_code):
        if abbreviation_blockette_number not in self.blockettes or \
                not self.blockettes[abbreviation_blockette_number]:
            raise ValueError("No blockettes %i available." %
                             abbreviation_blockette_number)

        if abbreviation_blockette_number == 31:
            key = "comment_code_key"
        elif abbreviation_blockette_number == 33:
            key = "abbreviation_lookup_code"
        elif abbreviation_blockette_number == 34:
            key = "unit_lookup_code"
        elif abbreviation_blockette_number in (41, 42, 43, 44, 45, 46, 47, 48,
                                               49):
            key = "response_lookup_key"
        else:
            raise NotImplementedError(str(abbreviation_blockette_number))

        blkts = [b for b in self.blockettes[abbreviation_blockette_number]
                 if int(getattr(b, key)) == int(lookup_code)]
        if len(blkts) > 1:
            msg = ("Found multiple blockettes %i with lookup code %i. Will "
                   "use the first one." % (abbreviation_blockette_number,
                                           lookup_code))
            warnings.warn(msg)
            blkts = blkts[:1]
        if not blkts:
            raise ValueError("Could not find a blockette %i with lookup code "
                             "%i." % (abbreviation_blockette_number,
                                      lookup_code))
        return blkts[0]

    def get_response_for_channel(self, blockettes_for_channel, epoch_str):
        """
        Create an ObsPy response object from all blockettes of a channel.

        This is a method instead of function as it needs to access the
        abbreviation dictionary.

        :param blockettes_for_channel: The blockettes for the channel to
            calculate the response for.
        :type blockettes_for_channel:
            List[:class:`~obspy.io.xseed.blockette.Blockette]
        :param epoch_str: A string representing the epoch. Used for nice
            warning and error message.
        :type epoch_str: str

        :rtype: :class:`obspy.core.inventory.response.Response`
        :returns: Inventory response object.
        """
        # Function generating more descriptive warning and error messages.
        def _epoch_warn_msg(msg):
            return "Epoch %s: %s" % (epoch_str, msg)

        # Return if there is not response.
        if set(_i.id for _i in blockettes_for_channel).issubset({52, 59}):
            return None

        transform_map = {'A': 'LAPLACE (RADIANS/SECOND)',
                         'B': 'LAPLACE (HERTZ)',
                         'D': 'DIGITAL (Z-TRANSFORM)'}

        transfer_map = {'A': 'ANALOG (RADIANS/SECOND',
                        'B': 'ANALOG (HERTZ)',
                        'D': 'DIGITAL'}

        # Parse blockette 60 and convert all the dictionary to their real
        # blockette counterparts. Who ever thought blockette 60 was a good
        # idea???
        _blockettes = []

        # Define the mappings for blockette 60. The key is the dictionary
        # blockette, the value a tuple of the corresponding blockette class and
        # another dictionary with the key being the name of the field in the
        # actual blockette mapped to the one in the dictionary. If a key is not
        # present, the same name is assumed.
        mappings = {
            41: (Blockette061, {
                "mappings": {
                    "number_of_coefficients": "number_of_factors"},
                "might_be_empty": ["FIR_coefficient"]
            }),
            43: (Blockette053, {
                "mappings": {
                    "transfer_function_types": "response_type"},
                "might_be_empty": ["real_zero", "real_pole", "imaginary_zero",
                                   "imaginary_pole", "real_zero_error",
                                   "real_pole_error", "imaginary_zero_error",
                                   "imaginary_pole_error"]
            }),
            44: (Blockette054, {
                "mappings": {},
                "might_be_empty": ["numerator_coefficient", "numerator_error",
                                   "denominator_coefficient",
                                   "denominator_error"]
            }),
            47: (Blockette057, {
                "mappings": {},
                "might_be_empty": []
            }),
            48: (Blockette058, {
                "mappings": {},
                "might_be_empty": ["sensitivity_for_calibration",
                                   "frequency_of_calibration_sensitivity",
                                   "time_of_above_calibration"]
            })
        }
        ignore_fields = ["stage_sequence_number", "response_name"]

        # Some preprocessing - mainly filter out comment blockettes and
        # translate all blockettes in blockette 60 to their full blockette
        # counterparts.
        for b in blockettes_for_channel:
            # Filter out comments.
            if b.id != 60:
                # Comments are irrelevant for the response.
                if b.id == 59:
                    continue
                _blockettes.append(b)
                continue
            # Translate blockette 60 to their true blockette counterparts.
            for _i, lookup_codes in enumerate(b.stages):
                stage_sequence_number = _i + 1
                for code in lookup_codes:
                    possible_dicts = []
                    for _j in (41, 42, 43, 44, 45, 46, 47, 48, 49):
                        try:
                            possible_dicts.append(
                                self.resolve_abbreviation(_j, code))
                        except ValueError:
                            continue
                    if not possible_dicts:
                        raise ValueError(_epoch_warn_msg(
                            "Failed to find dictionary response for key %i." %
                            code))
                    elif len(possible_dicts) > 1:
                        raise ValueError(_epoch_warn_msg(
                            "Found multiple dictionary responses for key %i." %
                            code))
                    _d = possible_dicts[0]
                    # Now it starts to get really ugly...
                    _m = mappings[_d.id]
                    _b = _m[0]()
                    _m = _m[1]

                    # Parse out the loop fields.
                    fields = []
                    for f in _b.get_fields():
                        if not isinstance(f, Loop):
                            fields.append(f)
                            continue
                        fields.extend(f.data_fields)

                    for field in fields:
                        if field.attribute_name in ignore_fields:
                            continue
                        elif field.attribute_name in _m["mappings"]:
                            key = _m["mappings"][field.attribute_name]
                        else:
                            key = field.attribute_name

                        # Some keys might not be set.
                        has_it = hasattr(_d, key)
                        if not has_it and key in _m["might_be_empty"]:
                            continue

                        setattr(_b, field.attribute_name, getattr(_d, key))
                    _b.stage_sequence_number = stage_sequence_number
                    _blockettes.append(_b)
        blockettes_for_channel = _blockettes

        # Get blockette 52.
        blkt52 = [_i for _i in blockettes_for_channel if _i.id == 52]
        if len(blkt52) != 1:
            raise InvalidResponseError("Must have exactly one blockette 52.")
        blkt52 = blkt52[0]

        # Sort the rest into stages.
        stages = collections.defaultdict(list)
        for b in blockettes_for_channel:
            # Blockette 52 does not belong to any particular stage.
            if b.id == 52:
                continue
            stages[b.stage_sequence_number].append(b)
        # Convert to a normal dictionary to not get any surprises like
        # automatically generated stages when testing if something is there.
        stages = dict(stages)

        # Get input units from blockette 52.
        if hasattr(blkt52, "units_of_signal_response"):
            try:
                input_units = self.resolve_abbreviation(
                    34, blkt52.units_of_signal_response)
            except ValueError:
                msg = ("Failed to resolve units of signal abbreviation in "
                       "blockette 52.")
                warnings.warn(_epoch_warn_msg(msg))
                input_units = None
        # Alternatively stage 0 might be blkt 62 which might also the full
        # units.
        else:
            if 0 in stages and stages[0] and stages[0][0].id == 62:
                try:
                    input_units = self.resolve_abbreviation(
                        34, stages[0][0].stage_signal_in_units)
                except ValueError:
                    msg = ("Failed to resolve units of signal abbreviation "
                           "in blockette 62.")
                    warnings.warn(_epoch_warn_msg(msg))
                    input_units = None
            else:
                input_units = None

        # Alternatively get them from the first stage that claims to have them.
        stage_x_output_units = None
        for _stage in list(range(1, max(stages.keys()) + 1)) + [0]:
            if _stage not in stages:
                continue
            _s = stages[_stage][0]
            for _attr in dir(_s):
                if "unit" in _attr and \
                        ("input" in _attr or "in unit" in _attr):
                    try:
                        stage_x_output_units = getattr(_s, _attr)
                        stage_x_output_units = \
                            self.resolve_abbreviation(
                                34, stage_x_output_units)
                    except ValueError:
                        pass
                    break

        # If both are None -> raise a warning.
        if input_units is None and stage_x_output_units is None:
            msg = "Could not determine input units."
            warnings.warn(_epoch_warn_msg(msg))
        # Prefer the input units from blockette 52, otherwise use the ones
        # from the first stage that has them.
        elif input_units is None:
            input_units = stage_x_output_units

        # If the input units are still unresolved for some reason, resolve
        # them now.
        if isinstance(input_units, int):
            try:
                input_units = self.resolve_abbreviation(34, input_units)
            except ValueError:
                msg = "Failed to resolve the input units abbreviation."
                warnings.warn(_epoch_warn_msg(msg))
                input_units = None

        # Find the output units by looping over the stages in reverse order
        # and finding the first stage whose first blockette has an output
        # unit set.
        unit_lookup_key = None
        for _s in reversed(sorted(stages.keys())):
            _s = stages[_s][0]
            for attr in dir(_s):
                if "unit" in attr and "out" in attr and "__" not in attr:
                    unit_lookup_key = getattr(_s, attr)
                    break
            else:
                continue
            break
        # Output units are the outputs of the final stage.
        if unit_lookup_key is None:
            msg = "Could not determine output units."
            warnings.warn(_epoch_warn_msg(msg))
            output_units = None
        else:
            try:
                output_units = self.resolve_abbreviation(34, unit_lookup_key)
            except ValueError:
                msg = "Could not resolve the output units abbreviation."
                warnings.warn(_epoch_warn_msg(msg))
                output_units = None

        # The remaining logic assumes that the stages list for blockette 0 at
        # least exists.
        if 0 not in stages:
            stages[0] = []

        # Stage zero usually only contains a single blockette (58 or 62).
        # Deal with cases where this is not true.
        if len(stages[0]) > 1:
            _blkts58 = [_i for _i in stages[0] if _i.id == 58]
            # Attempt to fix it - only works if there is a blockette 58 in
            # it and the rest of the stages are empty.
            if list(stages.keys()) == [0] and len(_blkts58) == 1:
                stages[1] = stages[0]
                stages[0] = _blkts58
            else:
                # Check if they are all identical - if they are, just choose
                # the first one.
                first = _blkts58[0]
                all_identical = True
                for _b in _blkts58[1:]:
                    if _b != first:
                        all_identical = False
                        break
                if not all_identical:
                    msg = ("Channel has multiple different blockettes "
                           "58 for stage 0. The last one will be chosen - "
                           "this is a faulty file - try to fix it!")
                    warnings.warn(_epoch_warn_msg(msg))
                    stages[0] = _blkts58[:-1]
                else:
                    msg = ("Channel has multiple (but identical) blockettes "
                           "58 for stage 0. Only one will be used.")
                    warnings.warn(_epoch_warn_msg(msg))
                    stages[0] = _blkts58[:1]

        # If there is no stage zero and exactly one other stages, use it to
        # reconstruct stage zero by just copying its stage 58.
        # 2nd condition: Make sure there is exactly one stage not zero.
        # 3rd condition: Make sure that stage has a blockette 58.
        if not len(stages[0]) and \
                len(set(stages.keys()).difference({0})) == 1 and \
                58 in [b.id for b in stages[sorted(stages.keys())[-1]]]:
            b = [b for b in stages[sorted(stages.keys())[-1]]
                 if b.id == 58][0]
            b = copy.deepcopy(b)
            b.stage_sequence_number = 0
            stages[0].append(b)

        # If still no stage 0, try to reconstruct it from all other
        # blockettes 58.
        if not stages[0]:
            _blkts58 = []
            for number in sorted(stages.keys()):
                _blkts58.extend([_i for _i in stages[number] if _i.id == 58])
            if not _blkts58:
                msg = "File has no stage 0 and no other blockettes 58. " \
                    "This is very likely just an invalid response."
                raise InvalidResponseError(msg)
            # Just multiply all gains - this appears to be what evalresp is
            # also doing.
            gain = 1.0
            for _b in _blkts58:
                if hasattr(_b, "sensitivity_gain") and \
                        _b.sensitivity_gain:
                    gain *= _b.sensitivity_gain
            stages[0] = [Blockette058()]
            # Use the last found frequency. If none is found, just set it to
            # one.
            all_frequencies = [_b.frequency for _b in _blkts58 if _b.frequency]
            if all_frequencies:
                stages[0][0].frequency = all_frequencies[-1]
            else:
                stages[0][0].frequency = 1.0
            stages[0][0].sensitivity_gain = gain
            stages[0][0].stage_sequency_number = 0
            stages[0][0].record_type = 'S'

        # The final stage 0 blockette must be a blockette 58 or 62.
        if stages[0][0].id not in (58, 62):
            msg = "Stage 0 must be a blockette 58 or 62."
            raise InvalidResponseError(msg)

        # Cannot have both, blockettes 53 and 54, in one stage.
        for stage, blkts in stages.items():
            _i = {b.id for b in blkts}.intersection({53, 54})
            if len(_i) > 1:
                msg = ("Stage %i has both, blockette 53 and 54. This is not "
                       "valid.") % stage
                raise InvalidResponseError(msg)

        # Assemble the total instrument sensitivity.
        if stages[0][0].id == 58:
            instrument_sensitivity = InstrumentSensitivity(
                value=stages[0][0].sensitivity_gain,
                frequency=stages[0][0].frequency,
                input_units=input_units.unit_name if input_units else None,
                output_units=output_units.unit_name
                if output_units is not None else None,
                input_units_description=input_units.unit_description
                if (input_units and hasattr(input_units, "unit_description"))
                else None,
                output_units_description=output_units.unit_description
                if (output_units and hasattr(output_units, "unit_description"))
                else None)
        # Does not exist if it is a single blockette 62.
        else:
            instrument_sensitivity = None

        # Trying to fit blockette 62 in stage 0 into the inventory object.
        if stages[0][0].id == 62:
            if len(stages[0]) != 1:
                msg = "If blockette 62 is in stage 0 it must be " \
                    "the only blockette in stage 0."
                raise InvalidResponseError(msg)
            # If blockette 62 is set for stage 0 and other stages are
            # present it should just be a summary of all the other stages
            # and can thus be removed.
            if len(stages.keys()) != 1:
                del stages[0]
            # Otherwise (62 is still stage 0, but not other stages exist) it
            # is the only stage describing the full response. Moving it to
            # stage 1 should trigger the rest of the logic to work correctly.
            else:
                # Just move it to stage 1 and it should be covered by the rest
                # of the logic.
                stages[1] = stages[0]
                del stages[0]

        # Afterwards loop over all other stages and assemble them in one list.
        response_stages = []
        for _i in sorted(set(stages.keys()).difference({0})):
            # Some SEED files have blockettes in the wrong order - sort them
            # to fix it.
            # The sorting is kind of awkward - essentially sort by id,
            # but make sure 57 + 58 are at the end.
            blkts = sorted(stages[_i], key=lambda x: int(x.blockette_id))

            b_a = [b_ for b_ in blkts if b_.blockette_id not in ("057", "058")]
            b_b57 = [b_ for b_ in blkts if b_.blockette_id in ("057")]
            b_b58 = [b_ for b_ in blkts if b_.blockette_id in ("058")]
            if len(b_b58) > 1:
                msg = ("Stage %i has %i blockettes 58. Only the last one "
                       "will be used." % (_i, len(b_b58)))
                warnings.warn(_epoch_warn_msg(msg))
                b_b58 = b_b58[-1:]
            blkts = b_a + b_b57 + b_b58

            # A bit undefined if it does not end with blockette 58 I think.
            # Not needed for blockette 62.
            if blkts[-1].id == 58:
                b58 = blkts[-1]
            elif blkts[-1].id != 58 and blkts[0].id != 62:
                msg = ("Response stage %i does not end with blockette 58. "
                       "Proceed at your own risk." % _i)
                warnings.warn(_epoch_warn_msg(msg))
                b58 = None
            else:
                b58 = None

            def _list(value):
                if hasattr(value, '__iter__'):
                    return value
                else:
                    return [value]

            # Poles and Zeros stage.
            if blkts[0].id == 53:
                # Must be 53 and 58.
                _blkt_set = {b_.id for b_ in blkts}
                if not _blkt_set.issubset({53, 57, 58}):
                    extra_blkts = _blkt_set.difference({53, 57, 58})
                    msg = "Stage %i: " \
                        "A stage with blockette 53 may only contain " \
                        "additional blockettes 57 and 58. This stage has " \
                        "the following additional blockettes: %s" % (
                            _i,
                            ", ".join(str(_i) for _i in sorted(extra_blkts)))
                    raise InvalidResponseError(msg)
                blkts53 = [b_ for b_ in blkts if b_.id == 53]
                blkts57 = [b_ for b_ in blkts if b_.id == 57]
                b53 = blkts53[-1]

                if len(blkts53) > 1:
                    msg = ("Stage %i has %i blockettes 53. Only the last one "
                           "will be used." % (_i, len(blkts53)))
                    warnings.warn(_epoch_warn_msg(msg))
                if len(blkts57) > 1:
                    msg = ("Stage %i has %i blockettes 57. Only the last one "
                           "will be used." % (_i, len(blkts57)))
                    warnings.warn(_epoch_warn_msg(msg))
                if blkts57:
                    b57 = blkts57[-1]
                else:
                    b57 = None

                zeros = []
                # Might not have zeros.
                if hasattr(b53, "real_zero"):
                    for r, i, r_err, i_err in zip(
                            _list(b53.real_zero), _list(b53.imaginary_zero),
                            _list(b53.real_zero_error),
                            _list(b53.imaginary_zero_error)):
                        z = ComplexWithUncertainties(r, i)
                        err = ComplexWithUncertainties(r_err, i_err)
                        z.lower_uncertainty = z - err
                        z.upper_uncertainty = z + err
                        zeros.append(z)
                poles = []
                # Might somehow also not have zeros.
                if hasattr(b53, "real_pole"):
                    for r, i, r_err, i_err in zip(
                            _list(b53.real_pole), _list(b53.imaginary_pole),
                            _list(b53.real_pole_error),
                            _list(b53.imaginary_pole_error)):
                        p = ComplexWithUncertainties(r, i)
                        err = ComplexWithUncertainties(r_err, i_err)
                        p.lower_uncertainty = p - err
                        p.upper_uncertainty = p + err
                        poles.append(p)

                try:
                    i_u = self.resolve_abbreviation(
                        34, b53.stage_signal_input_units)
                except ValueError:
                    msg = "Stage %i: Failed to resolve the stage signal " \
                        "input units abbreivation for blockette 53." % _i
                    warnings.warn(_epoch_warn_msg(msg))
                    i_u = None
                try:
                    o_u = self.resolve_abbreviation(
                        34, b53.stage_signal_output_units)
                except ValueError:
                    msg = "Stage %i: Failed to resolve the stage signal " \
                        "output units abbreviation for blockette 53." % _i
                    warnings.warn(_epoch_warn_msg(msg))
                    o_u = None

                response_stages.append(PolesZerosResponseStage(
                    stage_sequence_number=b53.stage_sequence_number,
                    stage_gain=getattr(b58, "sensitivity_gain", None)
                    if b58 else None,
                    stage_gain_frequency=getattr(b58, "frequency", None)
                    if b58 else None,
                    input_units=i_u.unit_name if i_u else None,
                    output_units=o_u.unit_name if o_u else None,
                    input_units_description=i_u.unit_description
                    if (i_u and hasattr(i_u, "unit_description")) else None,
                    output_units_description=o_u.unit_description
                    if (o_u and hasattr(o_u, "unit_description")) else None,
                    pz_transfer_function_type=transform_map[
                        b53.transfer_function_types],
                    normalization_frequency=b53.normalization_frequency,
                    zeros=zeros,
                    poles=poles,
                    normalization_factor=b53.A0_normalization_factor,
                    decimation_input_sample_rate=b57.input_sample_rate
                    if b57 else None,
                    decimation_factor=b57.decimation_factor
                    if b57 else None,
                    decimation_offset=b57.decimation_offset
                    if b57 else None,
                    decimation_delay=b57.estimated_delay
                    if b57 else None,
                    decimation_correction=b57.correction_applied
                    if b57 else None))
            # Response coefficients stage.
            elif blkts[0].id == 54:
                _blkts = [b_.id for b_ in blkts]
                if 57 not in _blkts:
                    msg = ("Stage %i: "
                           "Invalid response specification. A blockette 54 "
                           "must always be followed by a blockette 57 "
                           "which is missing.") % _i
                    raise InvalidResponseError(msg)
                # There can be multiple blockettes 54 in sequence in which
                # case numerators or denominators are chained from all of them.
                blkts54 = [b_ for b_ in blkts if b_.id == 54]
                blkts57 = [b_ for b_ in blkts if b_.id == 57]

                if len(blkts57) > 1:
                    msg = ("Stage %i has %i blockettes 57. Only the last one "
                           "will be used." % (_i, len(blkts57)))
                    warnings.warn(_epoch_warn_msg(msg))
                b57 = blkts57[-1]

                # Choose the first one as a reference.
                b54 = blkts54[0]
                # Use all of them for the coefficients.
                numerator = []
                denominator = []
                for b in blkts54:
                    if hasattr(b, "numerator_coefficient"):
                        _t = b.numerator_coefficient
                        if not hasattr(_t, "__iter__"):
                            _t = [_t]
                        numerator.extend(_t)
                    if hasattr(b, "denominator_coefficient"):
                        _t = b.denominator_coefficient
                        if not hasattr(_t, "__iter__"):
                            _t = [_t]
                        denominator.extend(_t)

                try:
                    i_u = self.resolve_abbreviation(
                        34, b54.signal_input_units)
                except ValueError:
                    msg = "Stage %i: Failed to resolve the signal input " \
                        "units abbreviation for blockette 54." % _i
                    warnings.warn(_epoch_warn_msg(msg))
                    i_u = None
                try:
                    o_u = self.resolve_abbreviation(
                        34, b54.signal_output_units)
                except ValueError:
                    msg = "Stage %i: Failed to resolve the signal output " \
                        "units abbreviation for blockette 54." % _i
                    warnings.warn(_epoch_warn_msg(msg))
                    o_u = None

                response_stages.append(CoefficientsTypeResponseStage(
                    stage_sequence_number=b54.stage_sequence_number,
                    stage_gain=b58.sensitivity_gain if b58 else None,
                    stage_gain_frequency=b58.frequency if b58 else None,
                    input_units=i_u.unit_name if i_u else None,
                    output_units=o_u.unit_name if o_u else None,
                    input_units_description=i_u.unit_description
                    if (i_u and hasattr(i_u, "unit_description")) else None,
                    output_units_description=o_u.unit_description
                    if (o_u and hasattr(o_u, "unit_description")) else None,
                    cf_transfer_function_type=transfer_map[b54.response_type],
                    numerator=numerator,
                    denominator=denominator,
                    decimation_input_sample_rate=b57.input_sample_rate,
                    decimation_factor=b57.decimation_factor,
                    decimation_offset=b57.decimation_offset,
                    decimation_delay=b57.estimated_delay,
                    decimation_correction=b57.correction_applied))
            # Response list stage.
            elif blkts[0].id == 55:
                assert set(b_.id for b_ in blkts).issubset({55, 57, 58})
                b57 = [_i for _i in blkts if _i.id == 57]
                if len(b57):
                    b57 = b57[0]
                else:
                    b57 = None

                b55 = blkts[0]
                i_u = self.resolve_abbreviation(
                    34, b55.stage_input_units)
                o_u = self.resolve_abbreviation(
                    34, b55.stage_output_units)
                response_list = [
                    ResponseListElement(f, a, p) for f, a, p in
                    zip(b55.frequency, b55.amplitude, b55.phase_angle)]
                response_stages.append(ResponseListResponseStage(
                    stage_sequence_number=b55.stage_sequence_number,
                    stage_gain=b58.sensitivity_gain if b58 else None,
                    stage_gain_frequency=b58.frequency if b58 else None,
                    input_units=i_u.unit_name,
                    output_units=o_u.unit_name,
                    input_units_description=i_u.unit_description
                    if hasattr(i_u, "unit_description") else None,
                    output_units_description=o_u.unit_description,
                    response_list_elements=response_list,
                    decimation_input_sample_rate=b57.input_sample_rate
                    if b57 else None,
                    decimation_factor=b57.decimation_factor if b57 else None,
                    decimation_offset=b57.decimation_offset if b57 else None,
                    decimation_delay=b57.estimated_delay if b57 else None,
                    decimation_correction=b57.correction_applied
                    if b57 else None))
            # Decimation stage.
            elif blkts[0].id == 57:
                if {b_.id for b_ in blkts} != {57, 58}:
                    msg = "Stage %i: A decimation stage with blockette 57 " \
                        "must be followed by a blockette 58 which is " \
                        "missing here." % _i
                    raise InvalidResponseError(msg)
                b57 = blkts[0]
                # Cannot assign units yet - will be added at the end in a
                # final pass by inferring it from the units of previous and
                # subsequent stages.
                response_stages.append(ResponseStage(
                    stage_sequence_number=b58.stage_sequence_number
                    if b58 else None,
                    stage_gain=b58.sensitivity_gain if b58 else None,
                    stage_gain_frequency=b58.frequency if b58 else None,
                    input_units="",
                    output_units="",
                    decimation_input_sample_rate=b57.input_sample_rate,
                    decimation_factor=b57.decimation_factor,
                    decimation_offset=b57.decimation_offset,
                    decimation_delay=b57.estimated_delay,
                    decimation_correction=b57.correction_applied))
            # Gain only stage.
            elif blkts[0].id == 58:
                assert [b_.id for b_ in blkts] == [58]
                # Cannot assign units yet - will be added at the end in a
                # final pass by inferring it from the units of previous and
                # subsequent stages.
                response_stages.append(ResponseStage(
                    stage_sequence_number=b58.stage_sequence_number
                    if b58 else None,
                    stage_gain=b58.sensitivity_gain if b58 else None,
                    stage_gain_frequency=b58.frequency if b58 else None,
                    input_units="",
                    output_units=""))
            # FIR stage.
            elif blkts[0].id == 61:
                _blkt_set = {b_.id for b_ in blkts}
                if not _blkt_set.issubset({61, 57, 58}):
                    extra_blkts = _blkt_set.difference({61, 57, 58})
                    msg = "Stage %i: " \
                        "A stage with blockette 61 may only contain " \
                        "additional blockettes 57 and 58. This stage has " \
                        "the following additional blockettes: %s" % (
                            _i,
                            ", ".join(str(_i) for _i in sorted(extra_blkts)))
                    raise InvalidResponseError(msg)

                blkts61 = [b_ for b_ in blkts if b_.id == 61]

                # Blockette 57.
                blkts57 = [b_ for b_ in blkts if b_.id == 57]
                if len(blkts57) > 1:
                    msg = ("Stage %i: "
                           "Multiple blockettes 57 found after blockette 61! "
                           "Will use the last one. This is an invalid file "
                           "- please check it!") % _i
                    warnings.warn(_epoch_warn_msg(msg))
                if blkts57:
                    b57 = blkts57[-1]
                else:
                    b57 = None

                # Use first blkt 61 as a reference.
                b61 = blkts61[0]

                # Use all of them for the coefficients.
                coefficients = []
                for b in blkts61:
                    if hasattr(b, "FIR_coefficient"):
                        _t = b.FIR_coefficient
                        if not hasattr(_t, "__iter__"):
                            _t = [_t]
                        coefficients.extend(_t)

                i_u = self.resolve_abbreviation(
                    34, b61.signal_in_units)
                o_u = self.resolve_abbreviation(
                    34, b61.signal_out_units)

                symmetry_map = {"A": "NONE", "B": "ODD", "C": "EVEN"}

                response_stages.append(FIRResponseStage(
                    stage_sequence_number=b61.stage_sequence_number,
                    stage_gain=b58.sensitivity_gain if b58 else None,
                    stage_gain_frequency=b58.frequency if b58 else None,
                    input_units=i_u.unit_name,
                    output_units=o_u.unit_name,
                    input_units_description=i_u.unit_description
                    if (i_u and hasattr(i_u, "unit_description")) else None,
                    output_units_description=o_u.unit_description
                    if (o_u and hasattr(o_u, "unit_description")) else None,
                    symmetry=symmetry_map[b61.symmetry_code],
                    coefficients=coefficients,
                    decimation_input_sample_rate=b57.input_sample_rate
                    if b57 else None,
                    decimation_factor=b57.decimation_factor
                    if b57 else None,
                    decimation_offset=b57.decimation_offset
                    if b57 else None,
                    decimation_delay=b57.estimated_delay
                    if b57 else None,
                    decimation_correction=b57.correction_applied
                    if b57 else None))
            elif blkts[0].id == 62:
                b62 = blkts[0]
                ids = {b_.id for b_ in blkts}
                assert ids.issubset({62, 57, 58})

                if 57 in ids:
                    b57 = [b_ for b_ in blkts if b_.id == 57][0]
                else:
                    b57 = None

                # Try to get the units.
                try:
                    i_u = self.resolve_abbreviation(
                        34, b62.stage_signal_in_units)
                except ValueError:
                    msg = "Stage %i: Failed to resolve the stage signal in " \
                          "units abbreivation for blockette 62." % _i
                    warnings.warn(_epoch_warn_msg(msg))
                    i_u = None
                try:
                    o_u = self.resolve_abbreviation(
                        34, b62.stage_signal_out_units)
                except ValueError:
                    msg = "Stage %i: Failed to resolve the stage signal out " \
                        "units abbreviation for blockette 62." % _i
                    warnings.warn(_epoch_warn_msg(msg))
                    o_u = None

                if getattr(b62, "polynomial_approximation_type", "M").upper() \
                        != "M":
                    msg = "Stage _i: Only the MACLAURIN polynomial " \
                        "approximation type is currently supported." % _i
                    raise InvalidResponseError(msg)

                # Get the coefficients.
                coefficients = []
                if hasattr(b62, "polynomial_coefficient"):
                    _t = b62.polynomial_coefficient
                    if not hasattr(_t, "__iter__"):
                        _t = [_t]
                    coefficients.extend(_t)

                response_stages.append(PolynomialResponseStage(
                    stage_sequence_number=b62.stage_sequence_number,
                    stage_gain=b58.sensitivity_gain if b58 else None,
                    stage_gain_frequency=b58.frequency if b58 else None,
                    input_units=i_u.unit_name if i_u else None,
                    output_units=o_u.unit_name if o_u else None,
                    input_units_description=i_u.unit_description
                    if (i_u and hasattr(i_u, "unit_description")) else None,
                    output_units_description=o_u.unit_description
                    if (o_u and hasattr(o_u, "unit_description")) else None,
                    frequency_lower_bound=getattr(
                        b62, "lower_valid_frequency_bound", None),
                    frequency_upper_bound=getattr(
                        b62, "upper_valid_frequency_bound", None),
                    # Always MACLAURIN - this is tested above.
                    approximation_type='MACLAURIN',
                    approximation_lower_bound=getattr(
                        b62, "lower_bound_of_approximation", None),
                    approximation_upper_bound=getattr(
                        b62, "upper_bound_of_approximation", None),
                    maximum_error=getattr(
                        b62, "maximum_absolute_error", None),
                    coefficients=coefficients,
                    decimation_input_sample_rate=b57.input_sample_rate
                    if b57 else None,
                    decimation_factor=b57.decimation_factor
                    if b57 else None,
                    decimation_offset=b57.decimation_offset
                    if b57 else None,
                    decimation_delay=b57.estimated_delay
                    if b57 else None,
                    decimation_correction=b57.correction_applied
                    if b57 else None))
            else:
                raise NotImplementedError(_epoch_warn_msg(
                    "Stage %i has the following blockettes: %s" % (
                        _i, ", ".join(b_.id for b_ in blkts))))

        # Do one last pass over the response stages to fill in missing units.
        for _i in range(len(response_stages)):
            s = response_stages[_i]
            if not s.input_units and _i != 0:
                s.input_units = response_stages[_i - 1].output_units
            if not s.output_units and (_i + 1) < len(response_stages):
                s.output_units = response_stages[_i + 1].input_units

        # If the first stage does not have an input unit but the instrument
        # sensitivity has, set that.
        if response_stages and \
                not getattr(response_stages[0], "input_units", None) and \
                getattr(instrument_sensitivity, "input_units", None):
            response_stages[0].input_units = instrument_sensitivity.input_units
            response_stages[0].input_units_description = \
                instrument_sensitivity.input_units_description

        # Create response object.
        return Response(instrument_sensitivity=instrument_sensitivity,
                        instrument_polynomial=None,
                        response_stages=response_stages)

    def _parse_seed(self, data):
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
        self._update_internal_seed_structure()

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
                    if isinstance(network_id, str):
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

    def _parse_xseed(self, data):
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
            self._parse_xml_blockette(headers[0].getchildren()[0], 'V',
                                      xseed_version))
        # Append all abbreviations.
        for blkt in headers[1].getchildren():
            self.temp['abbreviations'].append(
                self._parse_xml_blockette(blkt, 'A', xseed_version))
        # Append all stations.
        for control_header in headers[2:]:
            if not control_header.tag == 'station_control_header':
                continue
            self.temp['stations'].append([])
            for blkt in control_header.getchildren():
                self.temp['stations'][-1].append(
                    self._parse_xml_blockette(blkt, 'S', xseed_version))
        # Update internal values.
        self._update_internal_seed_structure()
        # Write the self.blockettes dictionary.
        for b in itertools.chain.from_iterable(self.stations +
                                               [self.abbreviations]):
            self.blockettes.setdefault(b.id, []).append(b)

    def _get_resp_string(self, resp, blockettes, station):
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
            '#\t\t=======================================\n').encode(
            'ascii', 'strict'))
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
                resp.write(blkt.get_resp(
                    station, channel_info['Channel'], self.abbreviations))
            except AttributeError:
                msg = 'RESP output for blockette %s not implemented yet.'
                raise AttributeError(msg % blkt.id)

    def _parse_xml_blockette(self, xml_blockette, record_type, xseed_version):
        """
        Takes the lxml tree of any blockette and returns a blockette object.
        """
        # Get blockette number.
        blockette_id = int(xml_blockette.values()[0])
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
            blockette_obj.parse_xml(xml_blockette)
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
            blockette_str = blockette_.get_seed()
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

    def _update_internal_seed_structure(self):
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
                except Exception:
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
            # remove spaces between blockettes. In some cases these might be
            # newlines.
            while data.read(1) in [b' ', b'\n']:
                continue
            data.seek(data.tell() - 1)
            try:
                blockette_id = int(data.read(3))
                blockette_length = int(data.read(4))
            except Exception:
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
                blockette_obj.parse_seed(data, blockette_length)
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

    def rotate_to_zne(self, stream):
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
        from obspy.signal.rotate import rotate2zne

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
        z, n, e = rotate2zne(*all_arguments)

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


def is_xseed(path_or_file_object):
    """
    Simple function checking if the passed object contains a XML-SEED file.
    Returns True of False. Only checks the name of the root tag, which should
    be "xseed".

    >>> from obspy.core.util import get_example_file
    >>> xseed_file = get_example_file("dataless.seed.BW_FURT.xml")
    >>> is_xseed(xseed_file)
    True
    >>> stationxml_file = get_example_file("IU_ANMO_00_BHZ.xml")
    >>> is_xseed(stationxml_file)
    False

    :param path_or_file_object: File name or file like object.
    """
    if isinstance(path_or_file_object, etree._Element):
        xmldoc = path_or_file_object
    else:
        try:
            xmldoc = etree.parse(path_or_file_object)
        except etree.XMLSyntaxError:
            return False
    try:
        root = xmldoc.getroot()
    except Exception:
        return False
    # check tag of root element
    try:
        assert root.tag == "xseed"
    except Exception:
        return False
    return True


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
