# -*- coding: utf-8 -*-

from cStringIO import StringIO
from protocol import ArcLinkClientFactory
from twisted.internet import reactor


class Client(object):
    """
    """
    def __init__(self, debug=True, host="webdc.eu", port=18001):
        self.debug = debug
        self.host = host
        self.port = port

    def _fetch(self, request_data, request_type, filename=None):
        if not filename:
            outfile = StringIO()
        else:
            outfile = open(filename, 'wb')
        factory = ArcLinkClientFactory(request_type=request_type,
                                       request_data=request_data,
                                       outfile=outfile, debug=self.debug)
        reactor.connectTCP(self.host, self.port, factory)
        reactor.run()

        print "MAEH"
        outfile.seek(0)
        data = outfile.read()
        outfile.close()
        return data

    def getWaveform(self):
        """
        """
        request_type = "WAVEFORM format=MSEED"
        request_data = ["2008,1,1,0,0,0 2008,1,1,0,1,0 BW ROTZ EHE .", ]
        return self._fetch(request_data, request_type)

    def saveWaveform(self, filename):
        """
        """
        request_type = "WAVEFORM format=MSEED"
        request_data = ["2008,1,1,0,0,0 2008,1,1,0,1,0 BW ROTZ EHE .", ]
        self._fetch(request_data, request_type)
