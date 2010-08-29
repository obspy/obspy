# -*- coding: utf-8 -*-

import os
import unittest
from lxml.etree import parse

from obspy.core.util import NamedTemporaryFile, path
from obspy.events import Event

class EventTestCase(unittest.TestCase):
    """
    Test suite for obspy.events.event.
    """

    def test_lockAndUnlockEvent(self):
        """
        Test locking and unlocking events.
        """
        e = Event()
        e.dummy_attribute = 1
        e._lock()
        e.dummy_attribute = 2
        e.dummy_attribute_2 = 3
        self.assertEqual(e.dummy_attribute, 1)
        self.assertFalse(hasattr(e, 'dummy_attribute_2'))
        e._unlock()
        e.dummy_attribute = 2
        e.dummy_attribute_2 = 3
        self.assertEqual(e.dummy_attribute, 2)
        self.assertTrue(hasattr(e, 'dummy_attribute_2'))

    def test_locateLockEvent(self):
        """
        Locating locks the event.
        """
        e = Event()
        e.dummy_attribute = 1
        e.locate()
        e.dummy_attribute = 2
        self.assertEqual(e.dummy_attribute, 1)

    def test_readWriteSeishub(self):
        """
        Read a seishub event, write it and compare with original
        """
        e = Event()
        #e.readSeishubXML(path("obspyck_20100826123745.xml")) # XXX path not working yet
        e.readSeishubXML("obspyck_20100826123745.xml")
        tempfile = NamedTemporaryFile().name
        e.writeSeishubXML(tempfile)
        #e.writeSeishubXML(path("obspyck_20100826123745.xml") + ".test") # XXX for debugging # XXX path not working yet
        #e.writeSeishubXML("obspyck_20100826123745.xml" + ".test") # XXX for debugging
        #xml1 = open(path("obspyck_20100826123745.xml")).read()
        xml1 = open("obspyck_20100826123745.xml").readlines() # XXX path not working yet
        xml2 = open(tempfile).readlines()
        os.remove(tempfile)
        # somehow the header line is differing (" vs. ') and an empty line follows, allow for that
        xml1.pop(0)
        xml1.pop(0)
        xml2.pop(0)
        for i, j in zip(xml1, xml2):
            self.assertEqual(i.strip(), j.strip())

def suite():
    return unittest.makeSuite(EventTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
