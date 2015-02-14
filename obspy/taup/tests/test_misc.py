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
            ('P', ['P', u'END']),
            ('S', ['S', u'END']),
            ('PPP', ['P', 'P', 'P', u'END']),
            ('PPPP', ['P', 'P', 'P', 'P', u'END']),
            ('SSSS', ['S', 'S', 'S', 'S', u'END']),
            ('Sn', ['Sn', u'END']),
            ('Pn', ['Pn', u'END']),
            ('Pg', ['Pg', u'END']),
            ('Sg', ['Sg', u'END']),
            ('Sb', ['Sb', u'END']),
            ('Pb', ['Pb', u'END']),
            ('PmP', ['P', 'm', 'P', u'END']),
            ('SmS', ['S', 'm', 'S', u'END']),
            ('PKP', ['P', 'K', 'P', u'END']),
            ('PKIKP', ['P', 'K', 'I', 'K', 'P', u'END']),
            ('SKS', ['S', 'K', 'S', u'END']),
            ('SKP', ['S', 'K', 'P', u'END']),
            ('PKS', ['P', 'K', 'S', u'END']),
            ('SKKS', ['S', 'K', 'K', 'S', u'END']),
            ('PKKP', ['P', 'K', 'K', 'P', u'END']),
            ('PKiKP', ['P', 'K', 'i', 'K', 'P', u'END']),
            ('PcP', ['P', 'c', 'P', u'END']),
            ('PmP', ['P', 'm', 'P', u'END']),
            ('ScS', ['S', 'c', 'S', u'END']),
            ('SKSSKS', ['S', 'K', 'S', 'S', 'K', 'S', u'END']),
            ('Pdiff', ['Pdiff', u'END']),
            ('Sdiff', ['Sdiff', u'END']),
            ('PS', ['P', 'S', u'END']),
            ('SP', ['S', 'P', u'END']),
            ('PmS', ['P', 'm', 'S', u'END']),
            ('SmP', ['S', 'm', 'P', u'END']),
            ('PcS', ['P', 'c', 'S', u'END']),
            ('ScP', ['S', 'c', 'P', u'END']),
            ('pP', ['p', 'P', u'END']),
            ('sS', ['s', 'S', u'END']),
            ('SSP', ['S', 'S', 'P', u'END']),
            ('PPS', ['P', 'P', 'S', u'END']),
            ('SKiKS', ['S', 'K', 'i', 'K', 'S', u'END']),
            ('SKJKP', ['S', 'K', 'J', 'K', 'P', u'END']),
            ('PKJKS', ['P', 'K', 'J', 'K', 'S', u'END']),
            ('PnPn', ['Pn', 'Pn', u'END']),
            ('SnSn', ['Sn', 'Sn', u'END']),
            ('Pvmp', ['P', 'vm', 'p', u'END']),
            ('PvmP', ['P', 'vm', 'P', u'END']),
            ('P410P', ['P', '410', 'P', u'END']),
            ('p^410P', ['p', '^410', 'P', u'END']),
            ('pv410P', ['p', 'v410', 'P', u'END']),
            ('P^mP', ['P', '^m', 'P', u'END']),
            ('PvmP', ['P', 'vm', 'P', u'END']),
            ('2kmps', ['2kmps', u'END']),
            ('22kmps', ['22kmps', u'END']),
            ('.2kmps', ['.2kmps', u'END']),
            ('23.kmps', ['23.kmps', u'END']),
            ('23.33kmps', ['23.33kmps', u'END']),
        ]

        for name, result in legs:
            self.assertEqual(leg_puller(name), result)


def suite():
    return unittest.makeSuite(TauPyMiscTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
