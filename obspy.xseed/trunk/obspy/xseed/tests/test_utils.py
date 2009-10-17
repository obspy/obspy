# -*- coding: utf-8 -*-

import datetime
from obspy.xseed import utils
import unittest


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
        self.assertEquals("hallo_welt", utils.toXMLTag(name))

    def test_DateTime2String(self):
        dt = datetime.datetime(2008, 12, 23, 01, 30, 22, 123456)
        self.assertEquals(utils.DateTime2String(dt), "2008,358,01:30:22.1234")
        dt = datetime.datetime(2008, 12, 23, 01, 30, 22, 98765)
        self.assertEquals(utils.DateTime2String(dt), "2008,358,01:30:22.0987")
        dt = datetime.datetime(2008, 12, 23, 01, 30, 22, 1234)
        self.assertEquals(utils.DateTime2String(dt), "2008,358,01:30:22.0012")
        dt = datetime.datetime(2008, 12, 23, 01, 30, 22, 123)
        self.assertEquals(utils.DateTime2String(dt), "2008,358,01:30:22.0001")
        dt = datetime.datetime(2008, 12, 23, 01, 30, 22, 9)
        self.assertEquals(utils.DateTime2String(dt), "2008,358,01:30:22.0000")
        dt = datetime.datetime(2008, 12, 23, 01, 30, 21)
        self.assertEquals(utils.DateTime2String(dt), "2008,358,01:30:21.0000")
        dt = datetime.datetime(2008, 12, 23, 01, 0, 0, 0)
        self.assertEquals(utils.DateTime2String(dt), "2008,358,01:00:00.0000")
        dt = datetime.date(2008, 12, 23)
        self.assertEquals(utils.DateTime2String(dt), "2008,358")

    def test_DateTime2StringCompact(self):
        dt = datetime.datetime(2008, 12, 23, 01, 30, 22, 123456)
        self.assertEquals(utils.DateTime2String(dt, True),
                          "2008,358,01:30:22.1234")
        dt = datetime.datetime(2008, 12, 23, 01, 30, 22)
        self.assertEquals(utils.DateTime2String(dt, True), "2008,358,01:30:22")
        dt = datetime.datetime(2008, 12, 23, 01, 30)
        self.assertEquals(utils.DateTime2String(dt, True), "2008,358,01:30")
        dt = datetime.datetime(2008, 12, 23, 01)
        self.assertEquals(utils.DateTime2String(dt, True), "2008,358,01")
        dt = datetime.datetime(2008, 12, 23)
        self.assertEquals(utils.DateTime2String(dt, True), "2008,358")

    def test_String2DateTime(self):
        st = "2008,358,01:30:22.0012"
        self.assertEquals(utils.String2DateTime(st),
                          datetime.datetime(2008, 12, 23, 01, 30, 22, 1200))
        st = "2008,358,01:30:22.0987"
        self.assertEquals(utils.String2DateTime(st),
                          datetime.datetime(2008, 12, 23, 01, 30, 22, 98700))
        st = "2008,358,01:30:22.9876"
        self.assertEquals(utils.String2DateTime(st),
                          datetime.datetime(2008, 12, 23, 01, 30, 22, 987600))
        st = "2008,358,01:30:22.0005"
        self.assertEquals(utils.String2DateTime(st),
                          datetime.datetime(2008, 12, 23, 01, 30, 22, 500))
        st = "2008,358,01:30:22.0000"
        self.assertEquals(utils.String2DateTime(st),
                          datetime.datetime(2008, 12, 23, 01, 30, 22, 0))
        st = "2008,358,01:30:22"
        self.assertEquals(utils.String2DateTime(st),
                          datetime.datetime(2008, 12, 23, 01, 30, 22, 0))
        st = "2008,358,01:30"
        self.assertEquals(utils.String2DateTime(st),
                          datetime.datetime(2008, 12, 23, 01, 30, 0, 0))
        st = "2008,358,01"
        self.assertEquals(utils.String2DateTime(st),
                          datetime.datetime(2008, 12, 23, 01, 0, 0, 0))
        st = "2008,358"
        self.assertEquals(utils.String2DateTime(st),
                          datetime.date(2008, 12, 23))
        st = "2008,358,01:30:22.5"
        try:
            utils.String2DateTime(st)
        except:
            pass


def suite():
    return unittest.makeSuite(UtilsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
