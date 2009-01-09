# -*- coding: utf-8 -*-

from StringIO import StringIO
from obspy.xseed.blockette.blockette010 import Blockette010
from obspy.xseed.blockette.blockette051 import Blockette051
from obspy.xseed.blockette.blockette054 import Blockette054
from obspy.xseed.parser import SEEDParser, SEEDParserException
import unittest


class ParserTestCase(unittest.TestCase):
    
    def test_invalidStartHeader(self):
        """
        A SEED Volume must start with a Volume Index Control Header.
        """
        data = "000001S 0510019~~0001000000"
        parser = SEEDParser(strict=True)
        self.assertRaises(SEEDParserException, parser.parse, StringIO(data))
    
    def test_invalidStartBlockette(self):
        """
        A SEED Volume must start with Blockette 010.
        """
        data = "000001V 0510019~~0001000000"
        parser = SEEDParser(strict=True)
        self.assertRaises(SEEDParserException, parser.parse, StringIO(data))
    
    def test_blocketteStartsAfterRecord(self):
        """
        '... 058003504 1.00000E+00 0.00000E+0000 000006S*0543864 ... '
        ' 0543864' -> results in Blockette 005
        """
        # create a valid blockette 010 with record length 256
        b010 = "0100018 2.408~~~~~"
        blockette = Blockette010(strict=True)
        blockette.parse(b010)
        self.assertEquals(b010, blockette.getSEEDString())
        # create a valid blockette 054
        b054 = "0540240A0400300300000009" + ("+1.58748E-03" * 18)
        blockette = Blockette054(strict=True)
        blockette.parse(b054)
        self.assertEquals(b054, blockette.getSEEDString())
        # combine data
        data  = "000001V " + b010 + (' ' * 230)
        data += "000002S " + b054 + (' ' * 8) 
        data += "000003S*" + b054 + (' ' * 8)
        # read records
        parser = SEEDParser(strict=True)
        parser.parse(StringIO(data))

    def test_multipleContinuedStationControlHeader(self):
        """
        """
        # create a valid blockette 010 with record length 256
        b010 = "0100018 2.408~~~~~"
        blockette = Blockette010(strict=True)
        blockette.parse(b010)
        self.assertEquals(b010, blockette.getSEEDString())
        # create a valid blockette 054
        b054 = "0540960A0400300300000039"
        nr = ""
        for i in range(0,78):
            nr = nr + "+1.000%02dE-03" % i # 960 chars
        blockette = Blockette054(strict=True)
        blockette.parse(b054 + nr)
        self.assertEquals(b054 + nr, blockette.getSEEDString())
        # create a blockette 052
        b051 = '05100271999,123~~0001000000'
        blockette = Blockette051(strict=True)
        blockette.parse(b051)
        # combine data (each line equals 256 chars)
        data  = "000001V " + b010 + (' ' * 230)
        data += "000002S " + b054 + nr[0:224] # 256-8-24 = 224
        data += "000003S*" + nr[224:472] # 256-8 = 248
        data += "000004S*" + nr[472:720] 
        data += "000005S*" + nr[720:] + b051 + ' '*5 # 5 spaces left
        self.assertEqual(len(data), 256*5)
        data += "000006S " + b054 + nr[0:224] # 256-8-24 = 224
        data += "000007S*" + nr[224:472] # 256-8 = 248
        data += "000008S*" + nr[472:720] 
        data += "000009S*" + nr[720:] + ' '*32 # 32 spaces left
        self.assertEqual(len(data), 256*9)
        # read records
        parser = SEEDParser(strict=True)
        parser.parse(StringIO(data))
        # check results
        self.assertEquals(sorted(parser.blockettes.keys()), [10, 51, 54])
        self.assertEquals(len(parser.blockettes[10]), 1)
        self.assertEquals(len(parser.blockettes[51]), 1)
        self.assertEquals(len(parser.blockettes[54]), 2)


def suite():
    return unittest.makeSuite(ParserTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
