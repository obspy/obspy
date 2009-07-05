# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette054
from obspy.xseed.blockette.blockette import BlocketteLengthException
import doctest
import os
import unittest


class BlocketteTestCase(unittest.TestCase):
    
    def test_invalidBlocketteLength(self):
        """
        A wrong blockette length should raise an exception.
        """
        # create a blockette 054 which is way to long
        b054 = "0540240A0400300300000020" + ("+1.58748E-03" * 40)
        blockette = Blockette054(strict = True)
        self.assertRaises(BlocketteLengthException, blockette.parse, b054)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(BlocketteTestCase, 'test'))
    docpath = 'doctests' + os.path.sep
    suite.addTest(doctest.DocFileSuite(docpath + 'blockette010.txt'))
    suite.addTest(doctest.DocFileSuite(docpath + 'blockette011.txt'))
    suite.addTest(doctest.DocFileSuite(docpath + 'blockette012.txt'))
    suite.addTest(doctest.DocFileSuite(docpath + 'blockette030.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette031.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette032.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette033.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette034.txt'))
    suite.addTest(doctest.DocFileSuite(docpath + 'blockette041.txt'))
    suite.addTest(doctest.DocFileSuite(docpath + 'blockette043.txt'))
    suite.addTest(doctest.DocFileSuite(docpath + 'blockette050.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette051.txt'))
    suite.addTest(doctest.DocFileSuite(docpath + 'blockette052.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette053.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette054.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette057.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette058.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette059.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette061.txt'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
