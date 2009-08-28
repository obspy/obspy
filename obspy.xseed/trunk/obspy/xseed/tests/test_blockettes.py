# -*- coding: utf-8 -*-

from glob import iglob
from obspy.xseed.blockette import Blockette054
from obspy.xseed.blockette.blockette import BlocketteLengthException
import doctest
import unittest


class BlocketteTestCase(unittest.TestCase):

    def test_invalidBlocketteLength(self):
        """
        A wrong blockette length should raise an exception.
        """
        # create a blockette 054 which is way to long
        b054 = "0540240A0400300300000020" + ("+1.58748E-03" * 40)
        blockette = Blockette054(strict=True)
        self.assertRaises(BlocketteLengthException, blockette.parseSEED, b054)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(BlocketteTestCase, 'test'))
    for file in iglob('doctests/*.txt'):
        suite.addTest(doctest.DocFileSuite(file))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
