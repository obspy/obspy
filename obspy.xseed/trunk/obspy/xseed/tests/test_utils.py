# -*- coding: utf-8 -*-

import unittest

from obspy.xseed import utils


class UtilsTestCase(unittest.TestCase):

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
    return unittest.makeSuite(UtilsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
