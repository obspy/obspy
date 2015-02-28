# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import socket


HOST = 'geofon.gfz-potsdam.de'    # The remote host
PORT = 18000              # The remote port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('s.connect((', HOST, PORT, '))')
s.connect((HOST, PORT))
sendbytes = b'HELLO\r'
print('Sent:', repr(sendbytes))
s.send(sendbytes)
data = s.recv(1024)
print('Received:', repr(data))
s.close()
