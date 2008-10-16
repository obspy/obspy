# -*- coding: utf-8 -*-

import unittest

from obspy.seed.parser import SEEDParser
from obspy.seed.blockette import Blockette054, Blockette010


class ParserTestCase(unittest.TestCase):

    def test_blocketteStartsAfterRecord(self):
        """
        '... 000006S*0543864...'
        '  *0543864'
        """
        # create a valid blockette 010 with record length 256
        b010 = "0100018 2.408~~~~~"
        blockette = Blockette010()
        blockette.parse(b010)
        self.assertEquals(b010, blockette.getSEEDString())
        
        # create a valid blockette 054
        b054 = "0540240A0400300300000009" + ("+1.58748E-03" * 18)
        blockette = Blockette054()
        blockette.parse(b054)
        self.assertEquals(b054, blockette.getSEEDString())
        
        # combine data
        data  = "000001V " + b010 + (' ' * 230)
        data += "000002S " + b054 + (' ' * 8) 
        data += "000003S*" + b054 + (' ' * 8)
        
        # read records
        parser = SEEDParser()
        parser.parse(data)


def suite():
    return unittest.makeSuite(ParserTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
