#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the high level obspy.taup.tau interface.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy.taup.seismic_phase import leg_puller


class TauPyMiscTestCase(unittest.TestCase):
    """
    Test suite for the things of TauPy that have no place elsewhere.
    """
    def test_leg_puller(self):
        """
        Tests the leg puller.
        """
        legs = [
            ('P', ['P', 'END']),
            ('S', ['S', 'END']),
            ('PPP', ['P', 'P', 'P', 'END']),
            ('PPPP', ['P', 'P', 'P', 'P', 'END']),
            ('SSSS', ['S', 'S', 'S', 'S', 'END']),
            ('Sn', ['Sn', 'END']),
            ('Pn', ['Pn', 'END']),
            ('Pg', ['Pg', 'END']),
            ('Sg', ['Sg', 'END']),
            ('Sb', ['Sb', 'END']),
            ('Pb', ['Pb', 'END']),
            ('PmP', ['P', 'm', 'P', 'END']),
            ('SmS', ['S', 'm', 'S', 'END']),
            ('PKP', ['P', 'K', 'P', 'END']),
            ('PKIKP', ['P', 'K', 'I', 'K', 'P', 'END']),
            ('SKS', ['S', 'K', 'S', 'END']),
            ('SKP', ['S', 'K', 'P', 'END']),
            ('PKS', ['P', 'K', 'S', 'END']),
            ('SKKS', ['S', 'K', 'K', 'S', 'END']),
            ('PKKP', ['P', 'K', 'K', 'P', 'END']),
            ('PKiKP', ['P', 'K', 'i', 'K', 'P', 'END']),
            ('PcP', ['P', 'c', 'P', 'END']),
            ('PmP', ['P', 'm', 'P', 'END']),
            ('ScS', ['S', 'c', 'S', 'END']),
            ('SKSSKS', ['S', 'K', 'S', 'S', 'K', 'S', 'END']),
            ('Pdiff', ['Pdiff', 'END']),
            ('Sdiff', ['Sdiff', 'END']),
            ('PS', ['P', 'S', 'END']),
            ('SP', ['S', 'P', 'END']),
            ('PmS', ['P', 'm', 'S', 'END']),
            ('SmP', ['S', 'm', 'P', 'END']),
            ('PcS', ['P', 'c', 'S', 'END']),
            ('ScP', ['S', 'c', 'P', 'END']),
            ('pP', ['p', 'P', 'END']),
            ('sS', ['s', 'S', 'END']),
            ('SSP', ['S', 'S', 'P', 'END']),
            ('PPS', ['P', 'P', 'S', 'END']),
            ('SKiKS', ['S', 'K', 'i', 'K', 'S', 'END']),
            ('SKJKP', ['S', 'K', 'J', 'K', 'P', 'END']),
            ('PKJKS', ['P', 'K', 'J', 'K', 'S', 'END']),
            ('PnPn', ['Pn', 'Pn', 'END']),
            ('SnSn', ['Sn', 'Sn', 'END']),
            ('Pvmp', ['P', 'vm', 'p', 'END']),
            ('PvmP', ['P', 'vm', 'P', 'END']),
            ('P410P', ['P', '410', 'P', 'END']),
            ('p^410P', ['p', '^410', 'P', 'END']),
            ('pv410P', ['p', 'v410', 'P', 'END']),
            ('P^mP', ['P', '^m', 'P', 'END']),
            ('PvmP', ['P', 'vm', 'P', 'END']),
            ('2kmps', ['2kmps', 'END']),
            ('22kmps', ['22kmps', 'END']),
            ('.2kmps', ['.2kmps', 'END']),
            ('23.kmps', ['23.kmps', 'END']),
            ('23.33kmps', ['23.33kmps', 'END']),
        ]

        for name, result in legs:
            self.assertEqual(leg_puller(name), result)


def suite():
    return unittest.makeSuite(TauPyMiscTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
