# -*- coding: utf-8 -*-

import unittest

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

def suite():
    return unittest.makeSuite(EventTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
