# -*- coding: utf-8 -*-

from obspy.core.util import add_doctests, add_unittests
import unittest


MODULE_NAME = "obspy.segy"


def suite():
    suite = unittest.TestSuite()
    add_doctests(suite, MODULE_NAME)
    add_unittests(suite, MODULE_NAME)
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
