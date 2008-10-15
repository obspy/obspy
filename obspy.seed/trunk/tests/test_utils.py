# -*- coding: utf-8 -*-

import unittest

from obspy.seed import utils


class UtilsTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_toAttribute(self):
        name = "Hallo Welt"
        self.assertEquals("hallo_welt", utils.toAttribute(name))

    def test_toXMLTag(self):
        name = "Hallo Welt"
        self.assertEquals("HalloWelt", utils.toXMLTag(name))

def suite():
    return unittest.makeSuite(UtilsTestSuite, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
