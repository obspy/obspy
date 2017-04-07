# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import io
import os

from .blockette import Blockette
from ..fields import FixedString, Float, Integer, Loop, VariableString
from ..utils import lookup_code, format_resp


class Blockette041(Blockette):
    """
    Blockette 041: FIR Dictionary Blockette.

    The FIR blockette is used to specify FIR (Finite Impulse Response)
    digital filter coefficients. It is an alternative to blockette [44] when
    specifying FIR filters. The blockette recognizes the various forms of
    filter symmetry and can exploit them to reduce the number of factors
    specified in the blockette. See Response (Coefficients) Blockette [54]
    for more information.
    """

    id = 41
    name = "FIR Dictionary"
    fields = [
        Integer(3, "Response Lookup Key", 4),
        VariableString(4, "Response Name", 1, 25, 'UN_'),
        FixedString(5, "Symmetry Code", 1, 'U'),
        Integer(6, "Signal In Units", 3, xpath=34),
        Integer(7, "Signal Out Units", 3, xpath=34),
        Integer(8, "Number of Factors", 4),
        # REPEAT field 9 for the Number of Factors
        Loop("FIR Coefficient", "Number of Factors", [
            Float(9, "FIR Coefficient", 14, mask='%+1.7e')], flat=True),
    ]

    def parse_seed(self, data, expected_length=0):
        """
        If number of FIR coefficients are larger than maximal blockette size of
        9999 chars a follow up blockette with the same blockette id and
        response lookup key is expected - this is checked here.
        """
        # convert to stream for test issues
        if isinstance(data, bytes):
            expected_length = len(data)
            data = io.BytesIO(data)
        elif isinstance(data, (str, native_str)):
            raise TypeError("Data must be bytes, not string")
        # get current lookup key
        pos = data.tell()
        data.read(7)
        global_lookup_key = int(data.read(4))
        data.seek(pos)
        # read first blockette
        temp = io.BytesIO()
        temp.write(data.read(expected_length))
        # check next blockettes
        while True:
            # save position
            pos = data.tell()
            try:
                blockette_id = int(data.read(3))
            except ValueError:
                break
            if blockette_id != 41:
                # different blockette id -> break
                break
            blockette_length = int(data.read(4))
            lookup_key = int(data.read(4))
            if lookup_key != global_lookup_key:
                # different lookup key -> break
                break
            # ok follow up blockette found - skip some unneeded fields
            self.fields[1].read(data)
            self.fields[2].read(data)
            self.fields[3].read(data)
            self.fields[4].read(data)
            self.fields[5].read(data)
            # remaining length in current blockette
            length = pos - data.tell() + blockette_length
            # read follow up blockette and append it to temporary blockette
            temp.write(data.read(length))
        # reposition file pointer
        data.seek(pos)
        # parse new combined temporary blockette
        temp.seek(0, os.SEEK_END)
        _len = temp.tell()
        temp.seek(0)
        Blockette.parse_seed(self, temp, expected_length=_len)

    def parse_xml(self, xml_doc, *args, **kwargs):
        if self.xseed_version == '1.0':
            xml_doc.find('fir_coefficient').tag = 'FIR_coefficient'
        Blockette.parse_xml(self, xml_doc, *args, **kwargs)

    def get_xml(self, *args, **kwargs):
        xml = Blockette.get_xml(self, *args, **kwargs)
        if self.xseed_version == '1.0':
            xml.find('FIR_coefficient').tag = 'fir_coefficient'
        return xml

    def get_resp(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        string = \
            '#\t\t+                     +--------------------------------+' + \
            '                      +\n' + \
            '#\t\t+                     |   FIR response,' + \
            '%6s ch %s   |                      +\n' % (station, channel) + \
            '#\t\t+                     +--------------------------------+' + \
            '                      +\n' + \
            '#\t\t\n' + \
            'B041F05     Symmetry type:                         %s\n' \
            % self.symmetry_code + \
            'B041F06     Response in units lookup:              %s - %s\n' \
            % (lookup_code(abbreviations, 34, 'unit_name',
                           'unit_lookup_code', self.signal_in_units),
               lookup_code(abbreviations, 34, 'unit_description',
                           'unit_lookup_code', self.signal_in_units)) + \
            'B041F07     Response out units lookup:             %s - %s\n' \
            % (lookup_code(abbreviations, 34, 'unit_name', 'unit_lookup_code',
                           self.signal_out_units),
               lookup_code(abbreviations, 34, 'unit_description',
                           'unit_lookup_code', self.signal_out_units)) + \
            'B041F08     Number of numerators:                  %s\n' \
            % self.number_of_factors

        if self.number_of_factors > 1:
            string += '#\t\tNumerator coefficients:\n' + \
                      '#\t\t  i, coefficient\n'
            for _i in range(self.number_of_factors):
                string += 'B041F09    %4s %13s\n' \
                    % (_i, format_resp(self.FIR_coefficient[_i], 6))
        elif self.number_of_factors == 1:
            string += '#\t\tNumerator coefficients:\n' + \
                '#\t\t  i, coefficient\n'
            string += 'B041F09    %4s %13s\n' \
                % (0, format_resp(self.FIR_coefficient, 6))
        string += '#\t\t\n'
        return string.encode()
