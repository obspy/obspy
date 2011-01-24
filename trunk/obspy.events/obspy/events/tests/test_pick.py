# -*- coding: utf-8 -*-

import unittest

from obspy.events import Pick

class PickTestCase(unittest.TestCase):
    """
    Test suite for obspy.events.pick.
    """

    def test_lockAndUnlockPick(self):
        """
        Test locking and unlocking picks.
        """
        p = Pick()
        p.dummy_attribute = 1
        p._lock()
        p.dummy_attribute = 2
        p.dummy_attribute_2 = 3
        self.assertEqual(p.dummy_attribute, 1)
        self.assertFalse(hasattr(p, 'dummy_attribute_2'))
        p._unlock()
        p.dummy_attribute = 2
        p.dummy_attribute_2 = 3
        self.assertEqual(p.dummy_attribute, 2)
        self.assertTrue(hasattr(p, 'dummy_attribute_2'))

def suite():
    return unittest.makeSuite(PickTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
