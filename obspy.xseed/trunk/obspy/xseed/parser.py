# -*- coding: utf-8 -*-

from StringIO import StringIO
from lxml.etree import Element, SubElement, tostring
from lxml.etree import parse as xmlparse
from obspy.xseed import blockette, utils
from obspy.xseed.blockette import Blockette011, Blockette012
from obspy.xseed.utils import SEEDtoRESPTime
import math
import os


CONTINUE_FROM_LAST_RECORD = '*'
HEADERS = ['V', 'A', 'S']
# @see: http://www.iris.edu/manuals/SEEDManual_V2.4.pdf, p. 22-24
HEADER_INFO = {
    'V': {'name': 'Volume Index Control Header',
          'blockettes': [10, 11, 12]},
    'A': {'name': 'Abbreviation Dictionary Control Header',
          'blockettes': [30, 31, 32, 33, 34, 41, 43, 44, 45, 46, 47, 48]},
    'S': {'name': 'Station Control Header',
          'blockettes': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]}
}
RESP_BLOCKETTES = [53, 54, 55, 56, 57, 58, 60, 61, 62]


class SEEDParserException(Exception):
    pass


class Parser(object):
    """
    The XML-SEED parser class parses dataless or full SEED volumes.
    
    The SEED file format description can be found at
    @see: http://www.iris.edu/manuals/SEEDManual_V2.4.pdf.
    
    The XML-SEED format was proposed in
    @see: http://www.orfeus-eu.org/Organization/Newsletter/vol6no2/xml.shtml,
    @see: http://www.jamstec.go.jp/pacific21/xmlninja/.
    """

    def __init__(self, debug=False, strict=False, compact=False):
        """
        Initializes the SEED parser.
        
        @type debug: Boolean.
        @param debug: Enables a verbose debug log during parsing of SEED file.
        @type strict: Boolean.
        @param strict: Parser will raise an exception if SEED files does not
            stay within the SEED specifications.
        @type compact: Boolean.
        @param compact: SEED volume will contain compact data strings. Missing
            time strings will be filled with 00:00:00.0000 if this option is
            disabled.
        """
        self.record_length = 4096
        self.version = 2.4
        self.blockettes = {}
        self.debug = debug
        self.strict = strict
        self.compact = compact
        # All parsed data is organized in volume, abbreviations and a list of
        # stations.
        self.volume = None
        self.abbreviations = None
        self.stations = []

    def read(self, data):
        """
        General parser method for XML-SEED and Dataless SEED files.
        
        @type data: Basestring or StringIO object.
        @param data: Filename or XSEED/SEED string as file pointer or StringIO.
        """
        # try to transform everything into StringIO object
        if isinstance(data, basestring):
            if os.path.isfile(data):
                # looks like a file - read it
                data = open(data, 'rb').read()
            # but could also be a big string with data
            data = StringIO(data)
        elif not hasattr(data, "read"):
            raise TypeError
        # check first byte of data StringIO object
        first_byte = data.read(1)
        data.seek(0)
        if first_byte.isdigit():
            # SEED volumes starts with a number
            self._parseSEED(data)
        elif first_byte == '<':
            # XML files should always starts with an '<'
            self._parseXSEED(data)
        else:
            raise TypeError

    def getXSEED(self, version='1.0'):
        """
        Returns a XML representation of all headers of a SEED volume.
        """
        if version not in ['1.0', '1.1']:
            raise SEEDParserException("Unknown XML-SEED version!")
        doc = Element("xseed", version=version)
        # Nothing to write if not all necessary data is available.
        if not self.volume or not self.abbreviations or \
                    len(self.stations) == 0:
            msg = 'No data to be written available.'
            raise SEEDParserException(msg)
        # Check blockettes:
        if not self._checkBlockettes():
            msg = 'Not all necessary blockettes are available.'
            raise SEEDParserException(msg)
        # All the following unfortunately is necessary to get a correct
        # Blockette 11:
        # Start with the station strings to be able to write Blockette 11
        # later on. The created list will contain lists with the first item
        # being the corresponding station identifier code and each part of the
        # record being a separate item.
        stations_list = []
        # Loop over all stations.
        for _i in self.stations:
            station = []
            # Blockette 50 always should be the first blockette
            station.append(_i[0].station_call_letters)
            # Loop over blockettes.
            station.extend(self._createCutAndFlushRecord(_i, 'S'))
            stations_list.append(station)
        # Make abbreviations.
        abbreviations = self._createCutAndFlushRecord(self.abbreviations, 'A')
        # Make volume string. To do this blockette 11 needs to be created.
        # It will be created first, than the whole volume  will be parsed
        # to a SEEDString. If the resulting string is longer than self.record
        # length -6 blockette 11 will be recreated with the newly won
        # informations.
        abbr_lenght = len(abbreviations)
        cur_count = 1 + abbr_lenght
        while True:
            blkt11 = Blockette011()
            blkt11.number_of_stations = len(self.stations)
            stations_lengths = [cur_count + 1]
            for _i in [len(_i) - 1 for _i in stations_list][:-1]:
                stations_lengths.append(stations_lengths[-1] + _i)
            blkt11.sequence_number_of_station_header = stations_lengths
            blkt11.station_identifier_code = \
                [_i[0].station_call_letters for _i in self.stations]
            self.volume.append(blkt11)
            # Blockette 12 is also needed.
            blkt12 = Blockette012()
            blkt12.number_of_spans_in_table = 0
            self.volume.append(blkt12)
            volume = self._createCutAndFlushRecord(self.volume, 'V')
            if cur_count - abbr_lenght < len(volume):
                cur_count += len(volume) - 1
                del self.volume[-1]
                del self.volume[-2]
                continue
            break
        # Now start actually filling the XML tree.
        # Volume header:
        root = SubElement(doc, utils.toXMLTag('Volume Index Control Header'))
        for blockette in self.volume:
            root.append(blockette.getXML(version=version))
        # Abbreviations:
        root = SubElement(doc,
                    utils.toXMLTag('Abbreviation Dictionary Control Header'))
        for blockette in self.abbreviations:
            root.append(blockette.getXML(version=version))
        # All blockettes for one station in one root element:
        for station in self.stations:
            root = SubElement(doc, utils.toXMLTag('Station Control Header'))
            for blockette in station:
                root.append(blockette.getXML(version=version))
        # To pass the XSD schema test an empty timespan control header is added
        # to the end of the file.
        root = SubElement(doc, utils.toXMLTag('Timespan Control Header'))
        # Also no data is present in all supported SEED files (for now).
        root = SubElement(doc, utils.toXMLTag('Data Records'))
        # Delete Blockettes 11 and 12.
        del self.volume[-1]
        del self.volume[-2]
        # Write XML String.
        return tostring(doc, pretty_print=True, xml_declaration=True,
                        encoding='utf-8')

    def writeXSEED(self, filename, *args, **kwargs):
        """
        Stores XML-SEED string into file with given name.
        """
        fh = open(filename, 'w')
        fh.write(self.getXSEED(*args, **kwargs))
        fh.close()

    def getSEED(self):
        """
        Takes everything stored in the object returns a valid SEED string.
        """
        # Nothing to write if not all necessary data is available.
        if not self.volume or not self.abbreviations or \
                    len(self.stations) == 0:
            msg = 'No data to be written available.'
            raise SEEDParserException(msg)
        # Check blockettes:
        if not self._checkBlockettes():
            msg = 'Not all necessary blockettes are available.'
            raise SEEDParserException(msg)
        # String to be written to:
        seed_string = ''
        # Start with the station strings to be able to write Blockette 11
        # later on. The created list will contain lists with the first item
        # being the corresponding station identifier code and each part of the
        # record being a separate item.
        stations_list = []
        # Loop over all stations.
        for _i in self.stations:
            station = []
            # Blockette 50 is always the first blockette
            station.append(_i[0].station_call_letters)
            # Loop over blockettes.
            station.extend(self._createCutAndFlushRecord(_i, 'S'))
            stations_list.append(station)
        # Create abbreviations.
        abbreviations = self._createCutAndFlushRecord(self.abbreviations, 'A')
        # Create volume string. Blockette 11 will be generated  first, than the 
        # whole volume will be parsed into a SEEDString. If the resulting 
        # string is longer than self.record length -6 blockette 11 will be 
        # recreated with the newly retrieved information.
        abbr_lenght = len(abbreviations)
        cur_count = 1 + abbr_lenght
        while True:
            blkt11 = Blockette011()
            blkt11.number_of_stations = len(self.stations)
            stations_lengths = [cur_count + 1]
            for _i in [len(_i) - 1 for _i in stations_list][:-1]:
                stations_lengths.append(stations_lengths[-1] + _i)
            blkt11.sequence_number_of_station_header = stations_lengths
            blkt11.station_identifier_code = \
                [_i[0].station_call_letters for _i in self.stations]
            self.volume.append(blkt11)
            volume = self._createCutAndFlushRecord(self.volume, 'V')
            if cur_count - abbr_lenght < len(volume):
                cur_count += len(volume) - 1
                del self.volume[-1]
                continue
            break
        cur_count = 1
        # Finally write the actual SEED String.
        for _i in volume:
            seed_string += '%06i' % cur_count + _i
            cur_count += 1
        # Delete Blockette 11 again.
        del self.volume[-1]
        for _i in abbreviations:
            seed_string += '%06i' % cur_count + _i
            cur_count += 1
        # Remove name of the stations.
        stations_list = [_i[1:] for _i in stations_list]
        for _i in stations_list:
            for _j in _i:
                seed_string += '%06i' % cur_count + _j
                cur_count += 1
        return seed_string

    def writeSEED(self, filename, *args, **kwargs):
        """
        Stores SEED string into file with given name.
        """
        fh = open(filename, 'wb')
        fh.write(self.getSEED(*args, **kwargs))
        fh.close()

    def getRESP(self):
        """
        Returns a SEED RESP file from the current obspy.xseed.Parser object.
        
        It aims to produce the same RESP files as when running rdseed with
        the command: "rdseed -f seed.test -R".
        """
        # Check if there are any stations at all.
        if len(self.stations) == 0:
            msg = 'No data to be written.'
            raise Exception(msg)
        # Channel Response list.
        resp_list = []
        # XXX: Need to check the available fields!
        # Loop over all stations.
        for station in self.stations:
            resp = StringIO('')
            blockettes = []
            # Read the current station information and store it.
            cur_station = station[0].station_call_letters.strip()
            cur_network = station[0].network_code.strip()
            # Loop over all blockettes in that station.
            for _i in xrange(1, len(station)):
                # Catch all blockette 52.
                if station[_i].id == 52:
                    cur_location = station[_i].location_identifier.strip()
                    cur_channel = station[_i].channel_identifier.strip()
                    # Take old list and send it to the resp parser.
                    if resp.len != 0:
                        # Send the blockettes to the parser and append to list.
                        self._getRESPString(resp, blockettes, cur_station)
                        resp_list.append([filename, resp])
                    # Create the filename.
                    filename = 'RESP.%s.%s.%s.%s' \
                        % (cur_network, cur_station, cur_location, cur_channel)
                    # Create new StringIO and list.
                    resp = StringIO('')
                    blockettes = []
                    blockettes.append(station[_i])
                    # Write header and the first two lines to the string.
                    header = \
                    '#\t\t<< obspy.xseed, Version 0.1.3 >>\n' + \
                    '#\t\t\n' + \
                    '#\t\t======== CHANNEL RESPONSE DATA ========\n' + \
                    'B050F03     Station:     %s\n' % cur_station + \
                    'B050F16     Network:     %s\n' % cur_network
                    # Write to StringIO.
                    resp.write(header)
                    continue
                blockettes.append(station[_i])
            # It might happen that no blockette 52 is specified,
            if len(blockettes) != 0:
                # One last time for the last channel.
                self._getRESPString(resp, blockettes, cur_station)
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
                for _i in xrange(1, len(channel_list)):
                    channel_list[_i][1].seek(0, 0)
                    channel_list[0][1].write(channel_list[_i][1].read())
                new_resp_list.append(channel_list[0])
        return new_resp_list

    def writeRESP(self, folder=''):
        """
        Stores channel responses into files within a given folder.
        """
        new_resp_list = self.getRESP()
        # Write the files.
        for response in new_resp_list:
            file = open(folder + os.sep + response[0], 'w')
            response[1].seek(0, 0)
            file.write(response[1].read())
            file.close()

    def _parseSEED(self, data):
        """
        Parses through a whole SEED volume. It will always parse the whole
        file and skip any time span data.
        
        @type data: File pointer or StringIO object.
        """
        # Jump to the beginning of the file.
        data.seek(0)
        # Retrieve some basic data like version and record length.
        temp = data.read(8)
        # Check whether it starts with record sequence number 1 and a volume
        # index control header.
        if temp != '000001V ':
            raise SEEDParserException("Expecting 000001V ")
        # The first blockette has to be Blockette 10.
        temp = data.read(3)
        if temp != '010':
            raise SEEDParserException("Expecting blockette 010")
        # Skip the next four bytes containing the length of the blockette.
        data.seek(4, 1)
        # Set the version.
        self.version = float(data.read(4))
        # Get the record length.
        length = pow(2, int(data.read(2)))
        # Test record length.
        data.seek(length)
        temp = data.read(6)
        if temp != '000002':
            msg = "Got an invalid logical record length %d" % length
            raise SEEDParserException(msg)
        self.record_length = length
        if self.debug:
            print "RECORD LENGTH:", self.record_length
        # Set all temporary attributes.
        self.temp = {'volume' : [], 'abbreviations' : [], 'stations' : []}
        # Jump back to beginning.
        data.seek(0)
        # Read the first record.
        record = data.read(self.record_length)
        merged_data = ''
        record_type = None
        # Loop through file and pass merged records to _parseMergedData.
        while record:
            record_continuation = (record[7] == CONTINUE_FROM_LAST_RECORD)
            same_record_type = (record[6] == record_type)
            if record_type == 'S' and record[8:11] != '050':
                record_continuation = True
            if record_continuation and same_record_type:
                # continued record
                merged_data += record[8:]
            else:
                self._parseMergedData(merged_data.strip(), record_type)
                # first or new type of record
                record_type = record[6]
                merged_data = record[8:]
                if record_type not in HEADERS:
                    # only parse headers, no data
                    merged_data = ''
                    record_type = None
                    break
            if self.debug:
                if not record_continuation:
                    print "========"
                print record[0:8]
            record = data.read(self.record_length)
        # Use parse once again.
        self._parseMergedData(merged_data.strip(), record_type)
        # Update the internal structure to finish parsing.
        self._updateInternalSEEDStructure()

    def _parseXSEED(self, data):
        """
        Parse a XML-SEED string.

        @type data: File pointer or StringIO object.
        """
        data.seek(0)
        headers = xmlparse(data).getroot().getchildren()
        # Set all temporary attributes.
        self.temp = {'volume' : [], 'abbreviations' : [], 'stations' : []}
        # Parse volume which is assumed to be the first header. Only parse
        # blockette 10 and discard the rest.
        self.temp['volume'].append(\
                    self._parseXMLBlockette(headers[0].getchildren()[0], 'V'))
        # Append all abbreviations.
        for blockette in headers[1].getchildren():
            self.temp['abbreviations'].append(\
                    self._parseXMLBlockette(blockette, 'A'))
        # Append all stations into seperate list items.
        for control_header in headers[2:]:
            if not control_header.tag == 'station_control_header':
                continue
            self.temp['stations'].append([])
            for blockette in control_header.getchildren():
                self.temp['stations'][-1].append(\
                                    self._parseXMLBlockette(blockette, 'S'))
        # Update internal values.
        self._updateInternalSEEDStructure()

    def _getRESPString(self, resp, blockettes, station):
        """
        Takes a file like object and a list of blockettes containing all
        blockettes for one channel and writes them RESP like to the StringIO.
        """
        blkt52 = blockettes[0]
        # The first blockette in the list always has to be Blockette 52.
        channel_info = {'Location' : blkt52.location_identifier,
                        'Channel' : blkt52.channel_identifier,
                        'Start date': blkt52.start_date,
                        'End date': blkt52.end_date}
        # Set location and end date default values or convert end time..
        if len(channel_info['Location']) == 0:
            channel_info['Location'] = '??'
        if len(channel_info['End date']) == 0:
            channel_info['End date'] = 'No Ending Time'
        else:
            channel_info['End date'] = SEEDtoRESPTime(channel_info['End date'])
        # Convert starttime.
        channel_info['Start date'] = SEEDtoRESPTime(channel_info['Start date'])
        # Write Blockette 52 stuff.
        resp.write(\
                'B052F03     Location:    %s\n' % channel_info['Location'] + \
                'B052F04     Channel:     %s\n' % channel_info['Channel'] + \
                'B052F22     Start date:  %s\n' % channel_info['Start date'] + \
                'B052F23     End date:    %s\n' % channel_info['End date'] + \
                '#\t\t=======================================\n')
        # Write all other blockettes. Currently now sorting takes place. This
        # is just an experiement to see how rdseed does it. The Blockettes
        # might need to be sorted.
        for blockette in blockettes[1:]:
            if blockette.id not in RESP_BLOCKETTES:
                continue
            #try:
            resp.write(blockette.getRESP(station, channel_info['Channel'],
                                             self.abbreviations))
            #except AttributeError:
            #    msg = 'RESP output for blockette %s not implemented yet.' \
            #                % blockette.id
            #    raise AttributeError(msg)

    def _parseXMLBlockette(self, XML_blockette, record_type):
        """
        Takes the lxml tree of any blockette and returns a blockette object.
        """
        # Get blockette number.
        blockette_id = int(XML_blockette.values()[0])
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
            blockette_obj.parseXML(XML_blockette)
            return blockette_obj
        elif blockette_id != 0:
            msg = "Unknown blockette type %d found" % blockette_id
            raise SEEDParserException(msg)

    def _createCutAndFlushRecord(self, blockettes, record_type):
        """
        Takes all blockettes of a record and return a list with finished
        records.
        
        If necessary it will cut the record and return two or more flushed
        records.
        
        The returned records also include the control header type code and the
        record continuation code. Therefore the returned record will have the
        lenght self.record_length - 6. Other methods are responsible for
        writing the sequence number.
        
        It will always return a list with records.
        """
        length = self.record_length - 8
        return_records = []
        # Loop over all blockettes.
        record = ''
        for blockette in blockettes:
            rec_len = len(record)
            # Never split a blockette’s “length/blockette type” section across
            # records.
            if rec_len + 7 > length:
                # Flush the rest of the record if necessary.
                record += ' ' * (length - rec_len)
                return_records.append(record)
                record = ''
                rec_len = 0
            blockette_str = blockette.getSEED()
            # Calculate how much of the blockette is too long.
            overhead = rec_len + len(blockette_str) - length
            # If negative overhead: Write blockette.
            if overhead <= 0:
                record += blockette_str
            # Otherwise finish the record and start one or more new ones.
            else:
                record += blockette_str[:len(blockette_str) - overhead]
                # The record so far not written.
                rest_of_the_record = blockette_str[(len(blockette_str) - \
                                                    overhead):]
                # Loop over the number of records to be written.
                for _i in xrange(int(math.ceil(len(rest_of_the_record) / \
                                                   float(length)))):
                    return_records.append(record)
                    record = ''
                    # It doesn't hurt to index a string more than its length.
                    record += rest_of_the_record[_i * length: (_i + 1) * length]
        if len(record) > 0:
            return_records.append(record)
        # Flush last record
        return_records[-1] = return_records[-1] + ' ' * \
                                            (length - len(return_records[-1]))
        # Add control header and continuation code.
        return_records[0] = record_type + ' ' + return_records[0]
        for _i in range(len(return_records) - 1):
            return_records[_i + 1] = record_type + '*' + return_records[_i + 1]
        return return_records

    def _checkBlockettes(self):
        """
        Checks if all blockettes necessary for creating a SEED String are
        available.
        
        Returns True/False.
        """
        if not 10 in [_i.id for _i in self.volume]:
            return False
        abb_blockettes = [_i.id for _i in self.abbreviations]
        if not 30 in abb_blockettes and not 33 in abb_blockettes and \
                        not 34 in abb_blockettes:
            return False
        # Check every station:
        for _i in self.stations:
            stat_blockettes = [_j.id for _j in _i]
            if not 50 in stat_blockettes and not 52 in stat_blockettes and \
                        not 58 in stat_blockettes:
                return False
        return True

    def _updateInternalSEEDStructure(self):
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
            # Delete Blockette 11.
            for _i in range(len(self.temp['volume']) - 1, 0, -1):
                if self.temp['volume'][_i].id == 11:
                    del self.temp['volume'][_i]
                    break
            self.volume = self.temp['volume']
            self.abbreviations = self.temp['abbreviations']
            self.stations.extend(self.temp['stations'])
            del self.temp
        else:
            msg = 'Merging not yet implemented.'
            raise NotImplementedError(msg)
        # Also make the version of the format 2.4.
        self.volume[0].version_of_format = 2.4

    def _parseMergedData(self, data, record_type):
        """
        This method takes any merged SEED record and writes its blockettes
        in the corresponding dictionary entry of self.temp.
        """
        if not data:
            return
        # Create StringIO for easier access.
        data = StringIO(data)
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
                msg = 'More than one Abbreviation Dictionary Control Headers' + \
                      ' found!'
                raise SEEDParserException(msg)
            root_attribute = self.temp['abbreviations']
        # Loop over all blockettes in data.
        while blockette_id != 0:
            # remove spaces between blockettes 
            while data.read(1) == ' ':
                continue
            data.seek(-1, 1)
            try:
                blockette_id = int(data.read(3))
                blockette_length = int(data.read(4))
            except:
                break
            data.seek(-7, 1)
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
                blockette_obj.parseSEED(data, blockette_length)
                root_attribute.append(blockette_obj)
                self.blockettes.setdefault(blockette_id, []).append(blockette_obj)
            elif blockette_id != 0:
                msg = "Unknown blockette type %d found" % blockette_id
                raise SEEDParserException(msg)
