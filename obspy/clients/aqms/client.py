"""
AQMS Wave Server client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & ISTI (Instrumental Software Technologies, Inc)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Obspy client that connects to AQMS proxy waveserver (PWS)
"""

import socket

from obspy.clients.aqms.waveserver import (get_ws_data,
                get_ws_samplerate,
                get_ws_channels,
                get_ws_times)

class Client(object):
    def __init__(self, host, port, timeout=None, debug=False):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.debug = debug
        self.socket = None

    def __enter__(self):
        self.socket = self.connect()
        return self

    def __exit__(self ,type, value, traceback):
        self.socket.close()

    def connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        return s

    def get_data(self, network="", station="", location="--",
                         channel="", start=None, end=None):

        response = get_ws_data( self.socket , network, 
                            station, channel, location, 
                            start, end , timeout=self.timeout)

        return response

    def get_samplerate(self, network="", station="", location="--",
                         channel=""):

        response = get_ws_samplerate( self.socket , network, 
                            station, channel, location )

        return response

    def get_times(self, network="", station="", location="--",
                         channel=""):

        response = get_ws_times( self.socket , network, 
                            station, channel, location )

        return response

    def get_channels( self ):
        response = get_ws_channels( self.socket )
        return response

if __name__ == '__main__':
    import time
    #with Client("aqmsdev1.isti.net", 9321, timeout=60) as client:
    with Client("ucbpp.geo.berkeley.edu", 9321, timeout=5) as client:
        start = time.time()-3600*4
        end = start + 600
        st = client.get_data(
                        network="HV", station="PAUD", channel="HHZ",
                        start = start, end=end)
        print(st)




