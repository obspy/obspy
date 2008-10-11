# -*- coding: utf-8 -*-

from StringIO import StringIO
from lxml.etree import Element, SubElement

from obspy.seed.types import IntegerSEEDField, FloatSEEDField 
from obspy.seed.types import StringSEEDField, VariableStringSEEDField
from obspy.seed.types import LoopingSEEDField, SimpleLoopingSEEDField


class SEEDParserBlocketteLengthException(Exception):
    pass


class Blockette:
    """General Blockette handling."""
    
    field01 = IntegerSEEDField("Blockette type", 3)
    field02 = IntegerSEEDField("Length of blockette", 4)
    
    def __init__(self, *args, **kwargs):
        self.debug = kwargs.get('debug', False)
        self.verify = kwargs.get('verify', True)
        self.strict = kwargs.get('strict', False)
        self.version = kwargs.get('version', 2.4)
        if self.debug:
            print "B%03d" % self.id
    
    def parse(self, data, expected_length=0):
        # for test issues
        if isinstance(data, basestring):
            expected_length = len(data)
            data = StringIO(data)
        start_pos = data.tell()
        if self.debug:
            temp=data.read(expected_length)
            print '  DATA:', temp
            data.seek(-expected_length,1)
        blockette_id = 'B%03d' % self.id
        blockette_name = self.name.title().replace(' ','')
        doc = Element(blockette_name, id=blockette_id)
        for i in range(1, 50):
            func_name = 'field%02d' % i
            # field does not exists -> break
            if not hasattr(self, func_name):
                break
            field = getattr(self, func_name)
            # blockette length reached -> break with warning, because fields 
            # still exist
            if data.tell()-start_pos >= expected_length:
                if not self.strict and not self.verify:
                    break
                if isinstance(field, LoopingSEEDField) or \
                   isinstance(field, SimpleLoopingSEEDField):
                    break
                msg = "End of blockette " + blockette_id + " reached " + \
                      "without parsing all expected fields"
                if self.strict:
                    raise SEEDParserBlocketteLengthException(msg)
                else:
                    print('WARN: ' + msg)
                break
            field_id = 'F%02d' % i
            if self.debug:
                print '  ' + field_id + ':',
            field_name = field.name.title().replace(' ','')
            result = field.read(data)
            if self.debug:
                print result
            if i==2:
                expected_length = result
            if isinstance(field, LoopingSEEDField):
                root = SubElement(doc, field_name)
                for stub in result:
                    item = SubElement(root, 'item')
                    for nodename, nodetext in stub.iteritems():
                        SubElement(item, nodename).text = unicode(nodetext)
            elif isinstance(field, SimpleLoopingSEEDField):
                root = SubElement(doc, field_name)
                for nodetext in result:
                    SubElement(root, 'item').text = unicode(nodetext)
            else:
                SubElement(doc, field_name, id=field_id).text = unicode(result)
        end_pos = data.tell()
        if self.verify or self.strict:
            blockette_length = end_pos-start_pos
            if expected_length != blockette_length:
                msg = 'Wrong size of Blockette %s (%d of %d)' % \
                      (blockette_id, blockette_length, expected_length)
                if self.strict:
                    raise SEEDParserBlocketteLengthException(msg)
                else:
                    print('WARN: ' + msg)
        return doc


class Blockette010(Blockette):
    """Blockette 010: Volume Identifier Blockette.
        
    This is the normal header blockette for station or event oriented network 
    volumes. Include it once at the beginning of each logical volume or sub-
    volume.
    
    Sample:
    010009502.1121992,001,00:00:00.0000~1992,002,00:00:00.0000~1993,029~
    IRIS _ DMC~Data for 1992,001~
    """
    
    id = 10
    name = "Volume Identifier Blockette"
    field03 = FloatSEEDField("Version of format", 4)
    field04 = IntegerSEEDField("Logical record length", 4)
    field05 = VariableStringSEEDField("Beginning time", 1, 22, 'T')
    field06 = VariableStringSEEDField("End time", 1, 22, 'T')
    
    def __init__(self, *args, **kwargs):
        Blockette.__init__(self, *args, **kwargs)
        if self.version>=2.3:
            self.field07 = VariableStringSEEDField("Volume Time", 1, 22, 'T')
            self.field08 = VariableStringSEEDField("Originating Organization", 
                                                   1, 80)
            self.field09 = VariableStringSEEDField("Label", 1, 80)


class Blockette011(Blockette):
    """Blockette 011: Volume Station Header Index Blockette.
    
    This is the index to the Station Identifier Blockettes [50] that appear 
    later in the volume. This blockette refers to each station described in 
    the station header section.
    
    Sample:
    0110054004AAK  000003ANMO 000007ANTO 000010BJI  000012
    """
    
    id = 11
    name = "Volume Station Header Index Blockette"
    field03 = IntegerSEEDField("Number of stations", 3)
    # REPEAT fields 4 — 5 for the Number of stations:
    _field04 = StringSEEDField("Station identifier code", 5)
    _field05 = IntegerSEEDField("Sequence number of station header", 6)
    field04 = LoopingSEEDField('Stations', field03, [_field04, _field05])


class Blockette030(Blockette):
    """Blockette 030: Data Format Dictionary Blockette.
    
    All volumes, with the exception of miniSEED data records, must have a Data 
    Format Dictionary Blockette [30]. Each Channel Identifier Blockette [52] 
    has a reference (field 16) back to a Data Format Dictionary Blockette 
    [30], so that SEED reading programs will know how to decode data for the 
    channels. Because every kind of data format requires an entry in the Data 
    Format Dictionary Blockette [30], each recording network needs to list 
    entries for each data format, if a heterogeneous mix of data formats are 
    included in a volume. This data format dictionary is used to decompress 
    the data correctly.
    
    Sample:
    0300086CDSNΔGain-RangedΔFormat~000200104M0~W2ΔD0-13ΔA-8191~D1415~
    P0:#0,1:#2,2:#4,3:#7~
    """
    
    id = 30
    name = "Data Format Dictionary Blockette"
    field03 = VariableStringSEEDField("Short descriptive name", 1, 50, 'UNLPS')
    field04 = IntegerSEEDField("Data format identifier code", 4)
    field05 = IntegerSEEDField("Data family type", 3)
    field06 = IntegerSEEDField("Number of decoder keys", 2)
    # REPEAT field 7 for the Number of decoder keys:
    _temp07 = VariableStringSEEDField("Decoder keys", flags='UNLPS')
    field07 = SimpleLoopingSEEDField(field06, _temp07)


class Blockette033(Blockette):
    """Blockette 033: Generic Abbreviation Blockette.
        
    Sample:
    0330055001(GSN)ΔGlobalΔSeismographΔNetworkΔ(IRIS/USGS)~
    """
    
    id= 33
    name = "Generic Abbreviation Blockette"
    field03 = IntegerSEEDField("Abbreviation lookup code", 3)
    field04 = VariableStringSEEDField("Abbreviation description", 1, 50, 'UNLPS')


class Blockette034(Blockette):
    """Blockette 034: Units Abbreviations Blockette.
    
    This blockette defines the units of measurement in a standard, repeatable 
    way. Mention each unit of measurement only once.
    
    Sample:
    0340044001M/S~VelocityΔinΔMetersΔPerΔSecond~
    """
    
    id = 34
    name = "Units Abbreviations Blockette"
    field03 = IntegerSEEDField("Unit lookup code", 3)
    field04 = VariableStringSEEDField("Unit name", 1, 20, 'UNP')
    field05 = VariableStringSEEDField("Unit description", 0, 50, 'UNLPS')


class Blockette050(Blockette):
    """Blockette 050: Station Identifier Blockette.
    
    Sample:
    0500097ANMO +34.946200-106.456700+1740.00006001Albuquerque, NewMexico, USA~
    0013210101989,241~~NIU
    """
    
    id = 50
    name = "Station Identifier Blockette"
    field03 = StringSEEDField("Station call letters", 5, 'UN')
    field04 = FloatSEEDField("Latitude", 10)
    field05 = FloatSEEDField("Longitude", 11)
    field06 = FloatSEEDField("Elevation", 7)
    field07 = IntegerSEEDField("Number of channels", 4)
    field08 = IntegerSEEDField("Number of station comments", 3)
    field09 = VariableStringSEEDField("Site name", 1, 60, 'UNLPS')
    field10 = IntegerSEEDField("Network identifier code", 3)
    field11 = IntegerSEEDField("word order 32 bit", 4)
    field12 = IntegerSEEDField("word order 16 bit", 2)
    field13 = VariableStringSEEDField("Start effective date", 1, 22, 'T')
    field14 = VariableStringSEEDField("End effective date", 0, 22, 'T')
    field15 = StringSEEDField("Update flag", 1)
    
    def __init__(self, *args, **kwargs):
        Blockette.__init__(self, *args, **kwargs)
        if self.version>=2.3:
            self.field16 = StringSEEDField("Network Code", 2, 'ULN')


class Blockette052(Blockette):
    """Blockette 052: Channel Identifier Blockette.
    
    Sample:
    0520119BHE0000004~001002+34.946200-106.456700+1740.0100.0090.0+00.0000112 
    2.000E+01 2.000E-030000CG~1991,042,20:48~~N
    """
    
    id = 52
    name = "Channel Identifier Blockette"
    field03 = StringSEEDField("Location identifier", 2, 'UN')
    field04 = StringSEEDField("Channel identifier", 3, 'UN')
    field05 = IntegerSEEDField("Subchannel identifier", 4)
    field06 = IntegerSEEDField("Instrument identifier", 3)
    field07 = VariableStringSEEDField("Optional comment", 0, 30, 'UNLPS')
    field08 = IntegerSEEDField("Units of signal response", 3)
    field09 = IntegerSEEDField("Units of calibration input", 3)
    field10 = FloatSEEDField("Latitude", 10)
    field11 = FloatSEEDField("Longitude", 11)
    field12 = FloatSEEDField("Elevation", 7)
    field13 = FloatSEEDField("Local depth", 5)
    field14 = FloatSEEDField("Azimuth", 5)
    field15 = FloatSEEDField("Dip", 5)
    field16 = IntegerSEEDField("Data format identifier code", 4)
    field17 = IntegerSEEDField("Data record length", 2)
    field18 = FloatSEEDField("Sample rate", 10)
    field19 = FloatSEEDField("Max clock drift", 10)
    field20 = IntegerSEEDField("Number of comments", 4)
    field21 = VariableStringSEEDField("Channel flags", 0, 26, 'U')
    field22 = VariableStringSEEDField("Start date", 1, 22, 'T')
    field23 = VariableStringSEEDField("End date", 0, 22, 'T')
    field24 = StringSEEDField("Update flag", 1)


class Blockette053(Blockette):
    """Blockette 053: Response (Poles & Zeros) Blockette.
    
    Sample:
    0530382B 1007008 7.87395E+00 5.00000E-02  3
     0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
     0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
    -1.27000E+01 0.00000E+00 0.00000E+00 0.00000E+00  4
    -1.96418E-03 1.96418E-03 0.00000E+00 0.00000E+00
    S-1.96418E-03-1.96418E-03 0.00000E+00 0.00000E+00
    53-6.23500E+00 7.81823E+00 0.00000E+00 0.00000E+00
    -6.23500E+00-7.81823E+00 0.00000E+00 0.00000E+00
    """
    
    id = 53
    name = "Response Poles and Zeros Blockette"
    
    field03 = StringSEEDField("Transfer function type", 1, 'U')
    field04 = IntegerSEEDField("Stage sequence number", 2)
    field05 = IntegerSEEDField("Stage signal input units", 3)
    field06 = IntegerSEEDField("Stage signal output units", 3)
    field07 = FloatSEEDField("AO normalization factor", 12)
    field08 = FloatSEEDField("Normalization frequency fn", 12)
    field09 = IntegerSEEDField("Number of complex zeros", 3)
    # REPEAT fields 10 — 13 for the Number of complex zeros:
    _temp10 = FloatSEEDField("Real zero", 12)
    _temp11 = FloatSEEDField("Imaginary zero", 12)
    _temp12 = FloatSEEDField("Real zero error", 12)
    _temp13 = FloatSEEDField("Imaginary zero error", 12)
    # Field 10 equals Field 10-13 in the SEED manual
    field10 = LoopingSEEDField('Complex zeros', field09, 
                               [_temp10, _temp11, _temp12, _temp13])
    # Field 11 equals Field 16 in the SEED manual
    field11 = IntegerSEEDField("Number of complex poles", 3)
    # REPEAT fields 15 — 18 for the Number of complex poles:
    _temp15 = FloatSEEDField("Real pole", 12)
    _temp16 = FloatSEEDField("Imaginary pole", 12)
    _temp17 = FloatSEEDField("Real pole error", 12)
    _temp18 = FloatSEEDField("Imaginary pole error", 12)
    # Field 12 equals Field 15-18 in the SEED manual
    field12 = LoopingSEEDField('Complex poles', field11, 
                               [_temp15, _temp16, _temp17, _temp18])


class Blockette054(Blockette):
    """Blockette 054: Response (Coefficients) Blockette.
    
    This blockette is usually used only for fi nite impulse response (FIR) 
    filter stages. You can express Laplace transforms this way, but you should 
    use the Response (Poles & Zeros) Blockettes [53] for this. You can express 
    IIR filters this way, but you should use the Response (Poles & Zeros) 
    Blockette [53] here, too, to avoid numerical stability problems. Usually, 
    you will follow this blockette with a Decimation Blockette [57] and a 
    Sensitivity/Gain Blockette [58] to complete the definition of the filter 
    stage.
    
    This blockette is the only blockette that might overflow the maximum 
    allowed value of 9,999 characters. If there are more coefficients than fit 
    in one record, list as many as will fit in the first occurrence of this 
    blockette (the counts of Number of numerators and Number of denominators 
    would then be set to the number included, not the total number). In the 
    next record, put the remaining number. Be sure to write and read these 
    blockettes in sequence, and be sure that the first few fields of both 
    records are identical. Reading (and writing) programs have to be able to 
    work with both blockettes as one after reading (or before writing). In 
    July 2007, the FDSN adopted a convention that requires the coefficients to 
    be listed in forward time order. As a reference, minimum-phase filters 
    (which are asymmetric) should be written with the largest values near the 
    beginning of the coeffi cient list.
    """
    
    id = 54
    name = "Response Coefficients Blockette"
    field03 = StringSEEDField("Response type", 1, 'U')
    field04 = IntegerSEEDField("Stage sequence number", 2)
    field05 = IntegerSEEDField("Signal input units", 3)
    field06 = IntegerSEEDField("Signal output units", 3)
    field07 = IntegerSEEDField("Number of numerators", 4)
    # REPEAT fields 8 — 9 for the Number of numerators:
    _temp08 = FloatSEEDField("Numerator coefficient", 12)
    _temp09 = FloatSEEDField("Numerator error", 12)
    # Field 08 equals Fields 08-09 in the SEED manual
    field08 = LoopingSEEDField('Numerator', field07, [_temp08, _temp09])
    # Field 09 equals Field 10 in the SEED manual
    field09 = IntegerSEEDField("Number of denominators", 4)
    # REPEAT fields 11 — 12 for the Number of denominators:
    _temp11 = FloatSEEDField("Denominator coefficient", 12)
    _temp12 = FloatSEEDField("Denominator error", 12)
    # Field 10 equals Fields 11-12 in the SEED manual
    field10 = LoopingSEEDField('Denominator', field09, [_temp11, _temp12])


class Blockette057(Blockette):
    """Blockette 057: Decimation Blockette.
    
    Many digital filtration schemes process a high sample rate data stream; 
    filter; then decimate, to produce the desired output. Use this blockette 
    to describe the decimation phase of the stage. You would usually place it 
    between a Response (Coefficients) Blockette [54] and the Sensitivity/Gain 
    Blockette [58] phases of the filtration stage of the channel. Include
    this blockette with non-decimated stages because you must still specify 
    the time delay. (In this case, the decimation factor is 1 and the offset 
    value is 0.)
    
    Sample:
    057005132 .0000E+02    1    0 0.0000E+00 0.0000E+00
    """
    
    id = 57
    name = "Decimation Blockette"
    field03 = IntegerSEEDField("Stage sequence number", 2)
    field04 = FloatSEEDField("Input sample rate", 10)
    field05 = IntegerSEEDField("Decimation factor", 5)
    field06 = IntegerSEEDField("Decimation offset", 5)
    field07 = FloatSEEDField("Estimated delay", 11)
    field08 = FloatSEEDField("Correction applied", 11)


class Blockette058(Blockette):
    """Blockette 058: Channel Sensitivity/Gain Blockette.
    
    When used as a gain (stage ≠ 0), this blockette is the gain for this stage
    at the given frequency. Different stages may be at different frequencies. 
    However, it is strongly recommended that the same frequency be used in all 
    stages of a cascade, if possible. When used as a sensitivity(stage=0), 
    this blockette is the sensitivity (in counts per ground motion) for the 
    entire channel at a given frequency, and is also referred to as the 
    overall gain. The frequency here may be different from the frequencies in 
    the gain specifications, but should be the same if possible. If you use 
    cascading (more than one filter stage), then SEED requires a gain for each 
    stage. A final sensitivity (Blockette [58], stage = 0, is required. If you 
    do not use cascading (only one stage), then SEED must see a gain, a 
    sensitivity, or both.
    
    Sample:
    0580035 3 3.27680E+03 0.00000E+00 0
    """
    
    id = 58
    name = "Channel Sensitivity or Gain Blockette"
    field03 = IntegerSEEDField("Stage sequence number", 2)
    field04 = FloatSEEDField("Sensitivity or gain", 12)
    field05 = FloatSEEDField("Frequency ", 12)
    field06 = IntegerSEEDField("Number of history values", 2)
    # REPEAT fields 7 — 9 for the Number of history values:
    _temp07 = FloatSEEDField("Sensitivity for calibration", 12)
    _temp08 = FloatSEEDField("Frequency of calibration", 12)
    _temp09 = VariableStringSEEDField("Time of above calibration", 1, 22 , 'T')
    # Field 07 equals Fields 07-09 in the SEED manual
    field07 = LoopingSEEDField('History values', field06, 
                               [_temp07, _temp08, _temp09])


class Blockette061(Blockette):
    """Blockette 061: FIR Response Blockette.
    
    The FIR blockette is used to specify FIR (Finite Impulse Response) digital 
    filter coefficients. It is an alternative to blockette [54] when 
    specifying FIR filters. The blockette recognizes the various forms of 
    filter symmetry and can exploit them to reduce the number of factors 
    specified to the blockette. In July 2007, the FDSN adopted a convention 
    that requires the coefficients to be listed in forward time order. 
    As a reference, minimum-phase filters (which are asymmetric) should be
    written with the largest values near the beginning of the coefficient list.
    """
    
    id = 61
    name = "FIR Response Blockette"
    field03 = IntegerSEEDField("Stage sequence number", 2)
    field04 = VariableStringSEEDField("Response Name", 1, 25, 'UN_')
    field05 = StringSEEDField("Symmetry Code", 1, 'U')
    field06 = IntegerSEEDField("Signal In Units", 3)
    field07 = IntegerSEEDField("Signal Out Units", 3)
    field08 = IntegerSEEDField("Number of Coefficients", 4)
    #REPEAT field 9 for the Number of Coefficients
    _temp09 = FloatSEEDField("FIR Coefficient", 14)
    field09 = SimpleLoopingSEEDField(field08, _temp09)


