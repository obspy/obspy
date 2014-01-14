# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from obspy.core.util import add_doctests, add_unittests
import unittest


MODULE_NAME = "obspy.seg2"


def suite():
    suite = unittest.TestSuite()
    add_doctests(suite, MODULE_NAME)
    add_unittests(suite, MODULE_NAME)
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
