# -*- coding: utf-8 -*-

import unittest
from obspy.core.util import add_unittests


MODULE_NAME = "obspy.taup"


def suite():
    suite = unittest.TestSuite()
    add_unittests(suite, MODULE_NAME)
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
