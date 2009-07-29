# -*- coding: utf-8 -*-

from cStringIO import StringIO
from twisted.internet import protocol, reactor
from twisted.protocols import basic
import time


class ArcLinkClientProtocol(basic.LineReceiver):
    """
    """
    def sendLine(self, line):
        """
        """
        if self.debug:
            print '>>> ' + line
        basic.LineReceiver.sendLine(self, line)

    def connectionMade(self):
        """
        """
        self.debug = self.factory.debug
        self.content = self.factory.outfile
        self.state = "HELLO"
        self.sendLine('HELLO')

    def lineReceived(self, line):
        if self.debug:
            print '... ' + line
        func = getattr(self, 'state_' + self.state)
        func(line)

    def state_HELLO(self, line):
        self.state = 'HELLO2'

    def state_HELLO2(self, line):
        self.state = 'USER'
        self.sendLine('USER ' + self.factory.user)

    def state_USER(self, line):
        if line != 'OK':
            raise Exception('USER: expected OK, got %s' % line)
        self.state = 'INSTITUTION'
        self.sendLine('INSTITUTION ' + self.factory.institution)

    def state_INSTITUTION(self, line):
        if line != 'OK':
            raise Exception('INSTITUTION: expected OK, got %s' % line)
        self.state = 'REQUEST'
        self.sendLine("REQUEST " + self.factory.request_type)
        for data in self.factory.request_data:
            self.sendLine(data)
        self.sendLine("END")

    def state_REQUEST(self, line):
        self.state = "STATUS"
        self.request_id = line
        self.sendLine("STATUS " + self.request_id)

    def state_STATUS(self, line):
        if not 'status="' in line:
            raise Exception('STATUS: expected XML, got %s' % line)
        if 'status="OK"' not in line:
            reactor.callLater(0, self.sendLine, "STATUS " + self.request_id)
            return
        self.state = "DOWNLOAD"
        self.sendLine("DOWNLOAD " + self.request_id)

    def state_DOWNLOAD(self, line):
        if line == 'ERROR':
            self.state = "STATUS"
            reactor.callLater(0, self.sendLine, "STATUS " + self.request_id)
            return
        self.state = "DOWNLOAD2"
        self.size = int(line)
        self.clock = time.clock()
        self.setRawMode()
        return

    def state_PURGE(self, line):
        if line != 'OK':
            raise Exception('PURGE: expected OK got %s' % line)
        self.sendLine("BYE")
        self.transport.loseConnection()

    def rawDataReceived(self, data):
        pos = self.content.tell()
        if pos + len(data) < self.size:
            self.content.write(data)
            if self.debug:
                print '... (data chunk)', pos, 'of', self.size
            return
        self.content.write(data[0:-5])
        self.setLineMode()
        if self.debug:
            print '... (data chunk)', self.content.tell(), 'of', self.size
            print '>>> END', '(%.3lf seconds)' % (time.clock() - self.clock)
        self.state = "PURGE"
        self.sendLine("PURGE " + self.request_id)
        return


class ArcLinkClientFactory(protocol.ClientFactory):
    """
    """
    protocol = ArcLinkClientProtocol

    def __init__(self, request_type, request_data=[], user="Anonymous",
                 institution="Anonymous", outfile=StringIO(), debug=False):
        self.user = user
        self.institution = institution
        self.request_type = request_type
        self.request_data = request_data
        self.outfile = outfile
        self.debug = debug

    def startedConnecting(self, connector):
        if self.debug:
            print "CONNECTED"

    def clientConnectionFailed(self, connector, reason):
        if self.debug:
            print 'connection failed:', reason.getErrorMessage()
        reactor.stop()
        reactor.crash()

    def clientConnectionLost(self, connector, reason):
        if self.debug:
            print 'connection lost:', reason.getErrorMessage()
        reactor.stop()
        reactor.crash()


if __name__ == '__main__':
    outfile = open('output.mseed', 'wb')
    factory = ArcLinkClientFactory(request_type="WAVEFORM format=MSEED",
        request_data=["2008,1,1,0,0,0 2008,1,1,0,1,0 BW ROTZ EHE .", ],
        outfile=outfile, debug=True)
    reactor.connectTCP('webdc.eu', 18001, factory)
    reactor.run()
    outfile.close()
