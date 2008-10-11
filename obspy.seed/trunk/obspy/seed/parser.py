# -*- coding: utf-8 -*-

from lxml.etree import Element, SubElement
from StringIO import StringIO

from obspy.seed import blockette


VOLUME_HEADER = 'V'
DICTIONARY_HEADER = 'A'
STATION_HEADER = 'S'
TIME_SPAN_HEADER = 'T'
CONTINUE_FROM_LAST_RECORD = '*'


class SEEDParserException(Exception):
    pass


class SEEDParser:
    """The SEED parser class generates a XML data format from a dataless or 
    full SEED file.
    
    The SEED file format description kann be found at
    @see: http://www.iris.edu/manuals/SEEDManual_V2.4.pdf
    """ 
    record_length = 4096
    version = None
    blockettes = {}
    
    def __init__(self, filename, verify=True, debug=False, strict=False):
        self.filename = filename
        self.debug = debug
        self.verify = verify
        self.strict = strict
        self.fp = open(filename)
        self.doc = Element("DatalessSEEDXML")
        if self.debug:
            print 'FILENAME:', filename
        self.parse()
    
    def __del__(self):
        self.fp.close()
    
    def parse(self):
        self.fp.seek(0)
        # retrieve some basic date, like version and record_length
        data = self.fp.read(8)
        if data!='000001V ':
            raise SEEDParserException("Expecting 000001V ")
        # B010 F01
        data = self.fp.read(3)
        if data!='010':
            raise SEEDParserException("Expecting blockette 010")
        # F02
        data = self.fp.read(4)
        # F03
        self.version = float(self.fp.read(4))
        # F04
        length = pow(2,int(self.fp.read(2))) 
        # FIX for some not standard conform SEED files which have an offset and
        # don't include the 8 bytes of the record identification block - so we 
        # just test
        self.fp.seek(length)
        data = self.fp.read(6)
        offset = 0
        if data!='000002':
            # ok not what we expected - try to add bytes to the record length
            length+=8
            offset=4
            self.fp.seek(length+offset)
            data = self.fp.read(6)
            if data!='000002':
                raise SEEDParserException("Got an invalid logical record " + \
                                          "length %d" % length)
            if self.verify:
                print "WARN: Not standard conform SEED volume - Volume " + \
                      "logical record length must be expressed as a power " + \
                      "of 2."
        self.record_length = length
        if self.debug:
            print "RECORD LENGTH:",self.record_length
        # jump back to beginning
        self.fp.seek(0)
        record = self.fp.read(self.record_length)
        data = ''
        control_header_type_code = None
        # loop through file
        while record:
            record_continuation = record[7] == CONTINUE_FROM_LAST_RECORD
            if record_continuation :
                # continued record
                data+=record[8:].strip()
            else:
                self.parseData(data, control_header_type_code)
                # first or new type of record
                control_header_type_code = record[6]
                data=record[8:]
                if control_header_type_code not in (VOLUME_HEADER, 
                                                    DICTIONARY_HEADER, 
                                                    STATION_HEADER):
                    # skip non wanted records
                    control_header_type_code = None
            if self.debug:
                if not record_continuation:
                    print "========"
                print record[0:8]
            record = self.fp.read(self.record_length)
        self.parseData(data, control_header_type_code)
    
    def parseData(self, text, control_header_type_code):
        data = StringIO(text)
        if not data:
            return
        if control_header_type_code == VOLUME_HEADER:
            self._parseVolumeIndexControlHeaders(data)
        elif control_header_type_code == DICTIONARY_HEADER:
            self._parseAbbreviationDictionaryControlHeaders(data)
#        elif control_header_type_code == STATION_HEADER:
#            self._parseStationControlHeaders(data)
    
    def _parseVolumeIndexControlHeaders(self, data):
        """Read and process the Volume Index Control Headers.
        
        Volume index control headers precede all data. Their primary purpose
        is to provide a directory to differentiate parts of the volume for 
        network and event distributions. Only field station volumes use Field 
        Volume Identifier Blockette [5].
        """
        root = SubElement(self.doc, "VolumeIndexControlHeaders")
        blockette_length = 0
        blockette_id = -1
        
        while blockette_id != 0:
            try:
                blockette_id = int(data.read(3))
                blockette_length = int(data.read(4))
            except:
                break
            data.seek(-7, 1)
            if blockette_id >= 10 and blockette_id<=12:
                class_name = 'Blockette%03d' % blockette_id
                if not hasattr(blockette, class_name):
                    raise SEEDParserException('Blockette %d not implemented!' %
                                              blockette_id)
                blockette_class = getattr(blockette, class_name)
                blockette_obj = blockette_class(debug = self.debug,
                                                verify = self.verify,
                                                strict = self.strict,
                                                version = self.version)
                root.append(blockette_obj.parse(data, blockette_length))
                self.blockettes.setdefault(blockette_id, []).append(blockette_obj)
            elif blockette_id != 0:
                raise SEEDParserException("Unknown blockette type %s " + \
                                          "found" % blockette_id)
    

    
    def _parseAbbreviationDictionaryControlHeaders(self, data):
        """Read and process the Abbreviation Dictionary Control Header.
        
        Dictionary records let you use abbreviations to refer to lengthy 
        descriptions without having to create external tables. Blockettes [43] 
        through [48] help reduce the amount of space used to specify intricate 
        channel responses in that you can write out the responses once, and 
        refer to them with short lookup codes, thereby eliminating the need to 
        repeat the same information; they are almost identical to blockettes 
        [53] through [58], but differ only in that they are set up for use as 
        response dictionary entries. Use them with the Response Reference 
        Blockette [60].
        """
        root = SubElement(self.doc, "AbbreviationDictionaryControlHeaders")
        blockette_length = 0
        blockette_id = -1
        
        while blockette_id != 0:
            try:
                blockette_id = int(data.read(3))
                blockette_length = int(data.read(4))
            except:
                break
            data.seek(-7, 1)
            if (blockette_id >= 30 and blockette_id<=35) or \
               (blockette_id >= 41 and blockette_id<=48):
                class_name = 'Blockette%03d' % blockette_id
                if not hasattr(blockette, class_name):
                    raise SEEDParserException('Blockette %d not implemented!' %
                                              blockette_id)
                blockette_class = getattr(blockette, class_name)
                blockette_obj = blockette_class(debug = self.debug,
                                                verify = self.verify,
                                                strict = self.strict,
                                                version = self.version)
                root.append(blockette_obj.parse(data, blockette_length))
                self.blockettes.setdefault(blockette_id, []).append(blockette_obj)
            elif blockette_id != 0:
                raise SEEDParserException("Unknown blockette type %s " + \
                                          "found" % blockette_id)
    
    
    def _parseStationControlHeaders(self, data):
        """Read and process the Station Control Headers.
        
        The station header records contain all the configuration and 
        identification information for the station and all its instruments.
        The SEED format provides a great deal of flexibility for associating 
        recording channels to the station, including the ability to support 
        different data formats dynamically. For each new station, start a new 
        logical record, set the remainder of any previous header records to 
        blanks, and write it out.
        
        For analog cascading, use the Response (Poles & Zeros) Blockette [53], 
        and the Channel Sensitivity/Gain Blockette [58] if needed. For digital 
        cascading, use the Response (Coefficients) Blockette [54], and the 
        Decimation Blockette [57] or Channel Sensitivity/Gain Blockette [58] 
        if needed. For additional documentation, you may also use the Response 
        List Blockette [55] or the Generic Response Blockette [56].
        """
        root = SubElement(self.doc, "StationControlHeaders")
        blockette_length = 0
        blockette_id = -1
        
        while blockette_id != 0:
            try:
                blockette_id = int(data.read(3))
                blockette_length = int(data.read(4))
            except:
                break
            data.seek(-7, 1)
            if blockette_id >= 50 and blockette_id<=62:
                class_name = 'Blockette%03d' % blockette_id
                if not hasattr(blockette, class_name):
                    raise SEEDParserException('Blockette %d not implemented!' %
                                              blockette_id)
                blockette_class = getattr(blockette, class_name)
                blockette_obj = blockette_class(debug = self.debug,
                                                verify = self.verify,
                                                strict = self.strict,
                                                version = self.version)
                root.append(blockette_obj.parse(data, blockette_length))
                self.blockettes.setdefault(blockette_id, []).append(blockette_obj)
            elif blockette_id != 0:
                raise SEEDParserException("Unknown blockette type %d found" %
                                          blockette_id)
