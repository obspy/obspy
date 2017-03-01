# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import os

from obspy.clients.nrl.client import NRL, LocalNRL, RemoteNRL


class NRLTestCase(unittest.TestCase):
    """
    NRL test suite.

    """
    def setUp(self):
        self.local_nrl_root = os.path.join(os.path.dirname(__file__),
                                           'data',
                                           'IRIS',
                                           )
        self._nrl_local = NRL(root=self.local_nrl_root)
        self._nrl_online = NRL(root='http://ds.iris.edu/NRL')
        self._nrl_default = NRL()

        # For Lloyd delete
        self._nrl_full_local = NRL(root=os.path.join(os.path.dirname(__file__),
                                           'data',
                                           'IRIS_full' ))

    def test_obj_types(self):
        self.assertIsInstance(self._nrl_local, NRL)
        self.assertIsInstance(self._nrl_local, LocalNRL)
        self.assertIsInstance(self._nrl_online, NRL)
        self.assertIsInstance(self._nrl_online, RemoteNRL)
        self.assertIsInstance(self._nrl_default, NRL)
        self.assertIsInstance(self._nrl_default, RemoteNRL)

    def test_get_response(self):
        nrl = self._nrl_full_local
        nrl.get_response


def suite():
    return unittest.makeSuite(NRLTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
