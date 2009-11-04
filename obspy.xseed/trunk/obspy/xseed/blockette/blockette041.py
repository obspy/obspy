# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette
from obspy.xseed.fields import Integer, VariableString, FixedString, Float, Loop
from obspy.xseed.utils import formatRESP, LookupCode


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
        Integer(6, "Signal In Units", 3, xpath = 34),
        Integer(7, "Signal Out Units", 3, xpath = 34),
        Integer(8, "Number of Factors", 4),
        #REPEAT field 9 for the Number of Factors
        Loop("FIR Coefficient", "Number of Factors", [
            Float(9, "FIR Coefficient", 14, mask='%+1.7e')], flat=True),
    ]

    def parseXML(self, xml_doc, *args, **kwargs):
        if self.XSEED_version == '1.0':
            xml_doc.find('fir_coefficient').tag = 'FIR_coefficient'
        Blockette.parseXML(self, xml_doc, *args, **kwargs)

    def getXML(self, *args, **kwargs):
        xml = Blockette.getXML(self, *args, **kwargs)
        if self.XSEED_version == '1.0':
            xml.find('FIR_coefficient').tag = 'fir_coefficient'
        return xml

    def getRESP(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        string = \
        '#\t\t+                     +--------------------------------+                      +\n' + \
        '#\t\t+                     |   FIR response,%6s ch %s   |                      +\n'\
                    %(station, channel) + \
        '#\t\t+                     +--------------------------------+                      +\n' + \
        '#\t\t\n' + \
        'B041F05     Symmetry type:                         %s\n' \
                % self.symmetry_code + \
        'B041F06     Response in units lookup:              %s - %s\n'\
            %(LookupCode(abbreviations, 34, 'unit_name', 'unit_lookup_code',
                         self.signal_in_units),
              LookupCode(abbreviations, 34, 'unit_description',
                    'unit_lookup_code', self.signal_in_units))  + \
        'B041F07     Response out units lookup:             %s - %s\n'\
            %(LookupCode(abbreviations, 34, 'unit_name', 'unit_lookup_code',
                         self.signal_out_units),
              LookupCode(abbreviations, 34, 'unit_description',
                    'unit_lookup_code', self.signal_out_units))  + \
        'B041F08     Number of numerators:                  %s\n' \
                % self.number_of_factors 
        if self.number_of_factors > 1:
            string +=  '#\t\tNumerator coefficients:\n' + \
                       '#\t\t  i, coefficient\n'
            for _i in xrange(self.number_of_factors):
                string += 'B041F09    %4s %13s\n' \
                            % (_i, formatRESP(self.FIR_coefficient[_i], 6))
        elif self.number_of_factors == 1:
            string +=  '#\t\tNumerator coefficients:\n' + \
                       '#\t\t  i, coefficient\n'
            string += 'B041F09    %4s %13s\n' \
                            % (0, formatRESP(self.FIR_coefficient, 6))
        string += '#\t\t\n'
        return string