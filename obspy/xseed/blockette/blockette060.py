# -*- coding: utf-8 -*-

from StringIO import StringIO
from lxml.etree import Element, SubElement
from obspy.xseed.blockette import Blockette
from obspy.xseed.fields import Integer, Loop
from obspy.xseed.utils import setXPath, getXPath
import sys


class Blockette060(Blockette):

    def __init__(self, *args, **kwargs):
        """
        """
        self.stages = []
        super(Blockette060, self).__init__()
    """
    Blockette 060: Response Reference Blockette.

    Use this blockette whenever you want to replace blockettes [53] through
    [58] and [61] with their dictionary counterparts, blockettes [43] through
    [48] and [41]. We recommend placing responses in stage order, even if this
    means using more than one Response Reference Blockette [60].

    Here is an example:
        Stage 1:    Response (Poles & Zeros) Blockette [53]
                    Channel Sensitivity/Gain Blockette [58]
                    First response reference blockette:
                    Response Reference Blockette [60]
        Stage 2:        [44] [47] [48]
        Stage 3:        [44] [47] [48]
        Stage 4:        [44] [47]
                        Channel Sensitivity/Gain Blockette [58]
        Stage 5:    Response (Coefficients) Blockette [54]
                    (End of first response reference blockette)
                    Second response reference blockette:
                    Response Reference Blockette [60]
        Stage 5         (continued): [47] [48]
        Stage 6:        [44] [47] [48]
                    (End of second response reference blockette)

    Substitute Response Reference Blockette [60] anywhere the original
    blockette would go, but be sure to place it in the same position as the
    original would have gone. (Note that this blockette uses a repeating field
    (response reference) within another repeating field (stage value). This is
    the only blockette in the current version (2.1) that has this "two
    dimensional" structure.)
    """

    id = 60
    name = "Response Reference Blockette"
    fields = [
        Integer(3, "Number of stages", 2),
        #REPEAT field 4, with appropriate fields 5 and 6, for each filter stage
        Loop("FIR Coefficient", "Number of stages", [
            Integer(4, "Stage sequence number", 2),
            Integer(5, "Number of responses", 2),
            #REPEAT field 6, one for each response within each stage:
            Loop("Response lookup key", "Number of responses", [
                Integer(6, "Response lookup key", 4)], omit_tag=True),
        ]),
    ]

    def parseSEED(self, data, length=0, *args, **kwargs):
        """
        Read Blockette 60.
        """
        # convert to stream for test issues
        if isinstance(data, basestring):
            length = len(data)
            data = StringIO(data)
        new_data = data.read(length)
        new_data = new_data[7:]
        number_of_stages = int(new_data[0:2])
        # Loop over all stages.
        counter = 2
        for _i in xrange(number_of_stages):
            number_of_responses = int(new_data[counter + 2:counter + 4])
            self.stages.append([])
            # Start inner loop
            counter += 4
            for _j in xrange(number_of_responses):
                # Append to last list.
                self.stages[-1].append(int(new_data[counter:counter + 4]))
                counter += 4

    def getSEED(self, *args, **kwargs):
        """
        Writes Blockette 60.
        """
        data = ''
        # Write number of stages.
        data += '%2d' % len(self.stages)
        # Loop over all items in self.stages.
        stage_number = 1
        for stage in self.stages:
            # Write stage sequence number.
            data += '%2d' % stage_number
            stage_number += 1
            # Write number of items.
            data += '%2d' % len(stage)
            for number in stage:
                data += '%4d' % number
        # Add header.
        length = len(data) + 7
        header = '060%4d' % length
        data = header + data
        return data

    def getXML(self, xseed_version, *args, **kwargs):
        """
        Write XML.
        """
        if xseed_version == '1.0':
            msg = 'The xsd-validation file for XML-SEED version 1.0 does ' + \
                  'not support Blockette 60. It will be written but ' + \
                  'please be aware that the file cannot be validated.\n' + \
                  'If you want to validate your file please use XSEED ' + \
                  'version 1.1.\n'
            sys.stdout.write(msg)
        node = Element('response_reference', blockette="060")
        SubElement(node, 'number_of_stages').text = str(len(self.stages))
        # Loop over stages.
        for _i in xrange(len(self.stages)):
            inner_stage = SubElement(node, 'stage')
            SubElement(inner_stage, 'stage_sequence_number').text = str(_i + 1)
            SubElement(inner_stage, 'number_of_responses').text = \
                                                    str(len(self.stages[_i]))
            for _j in xrange(len(self.stages[_i])):
                SubElement(inner_stage, 'response_lookup_key').text = \
                            setXPath('dictionary', self.stages[_i][_j])
        return node

    def parseXML(self, xml_doc, version='1.0', *args, **kwargs):
        """
        Read XML of blockette 60.
        """
        # Loop over ch
        for child in xml_doc.getchildren():
            if child.tag != 'stage':
                continue
            self.stages.append([])
            for inner_child in child.getchildren():
                if inner_child.tag != 'response_lookup_key':
                    continue
                # for legacy support meaning XSEED without XPaths.:
                if inner_child.text.isdigit():
                    self.stages[-1].append(int(inner_child.text))
                else:
                    self.stages[-1].append(getXPath(inner_child.text))

    def getRESP(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        string = ''
        # Possible dictionary blockettes.
        dict_blockettes = [41, 43, 44, 45, 46, 47, 48]
        for _i in xrange(len(self.stages)):
            string += \
            '#\t\t+            +----------------------------------' + \
            '----------------+             +\n' + \
            '#\t\t+            |   Response Reference Information,' + \
            '%6s ch %s   |             +\n' % (station, channel) + \
            '#\t\t+            +----------------------------------' + \
            '----------------+             +\n' + \
            '#\t\t\n' + \
            'B060F03     Number of Stages:                      %s\n' \
                    % len(self.stages) + \
            'B060F04     Stage number:                          %s\n' \
                % (_i + 1) + \
            'B060F05     Number of Responses:                   %s\n' \
                % len(self.stages[_i]) + \
            '#\t\t\n'
            # Loop over all keys and print the information in order.
            for response_key in self.stages[_i]:
                # Find the corresponding key in the abbreviations.
                found_abbrev = False
                for blockette in abbreviations:
                    if blockette.id in dict_blockettes and \
                                blockette.response_lookup_key == response_key:
                        try:
                            string += \
                                blockette.getRESP(station, channel,
                                                  abbreviations)
                            found_abbrev = True
                        except AttributeError:
                            msg = 'RESP output not implemented for ' + \
                                  'blockette %d.' % blockette.id
                            raise AttributeError(msg)
                if not found_abbrev:
                    msg = 'The reference blockette for response key ' + \
                          '%d could not be found.' % response_key
                    raise Exception(msg)
        string += '#\t\t\n'
        return string
