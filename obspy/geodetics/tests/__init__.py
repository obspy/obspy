# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy.core.util import add_unittests, add_doctests


MODULE_NAME = "obspy.geodetics"


def suite():
    suite = unittest.TestSuite()
    add_unittests(suite, MODULE_NAME)
    add_doctests(suite, MODULE_NAME)
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
