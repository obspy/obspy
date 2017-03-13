# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import socket
import unittest

from vcr import vcr


class SocketTestCase(unittest.TestCase):
    """
    Test suite using socket
    """
    @vcr
    def test_seedlink(self):
        s = socket.socket()
        s.connect(('geofon.gfz-potsdam.de', 18000))
        s.send(b'HELLO\r\n')
        data = s.recv(1024)
        s.close()
        self.assertIn(b'SeedLink', data)

    @vcr
    def test_arclink(self):
        s = socket.socket()
        s.connect(('webdc.eu', 18001))
        s.send(b'HELLO\r\n')
        data = s.recv(1024)
        s.close()
        self.assertIn(b'ArcLink', data)


if __name__ == '__main__':
    unittest.main()
