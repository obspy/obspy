# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import os

from obspy.io.resp.nrl import NRL
from obspy.io.resp.parser import read_resp


class ParserTestCase(unittest.TestCase):
    """
    RESP Parser test suite

    """
    pass

def suite():
    return unittest.makeSuite(ParserTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
