#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the StationJSON writer.

:copyright:
    Mathijs Koymans (koymans@knmi.nl, 2016)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

class StationJSONTestCase(unittest.TestCase):
        """
        Skip tests
        """
        self.assertTrue(True)


def suite():
    return unittest.makeSuite(StationJSONTestCase, "test")


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
