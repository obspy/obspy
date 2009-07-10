#!/usr/bin/python
# -*- coding: utf-8 -*-

from twisted.internet import protocol, reactor
from twisted.protocols import basic
import time


HELLO = 0
HELLO2 = 1
USER = 2
INSTITUTION = 3
REQUEST = 4
STATUS = 5
DOWNLOAD = 6
DOWNLOAD2 = 7
PURGE = 8
BYE = 9


class ArcLinkClientProtocol(basic.LineReceiver):
    """
    """
    def sendLine(self, line):
        """
        """
        if self.factory.debug:
            print '>>> ' + line
        basic.LineReceiver.sendLine(self, line)

    def connectionMade(self):
        """
        """
        self.content = open(self.factory.outfile, 'wb')
        self.state = HELLO
        self.sendLine('HELLO')

    def lineReceived(self, line):
        if self.factory.debug:
            print '... ' + line
        if self.state == HELLO:
            self.state = HELLO2
            return
        elif self.state == HELLO2:
            self.state = USER
            self.sendLine('USER ' + self.factory.user)
            return
        elif self.state == USER:
            if line != 'OK':
                raise Exception('USER: expected OK, got %s' % line)
            self.state = INSTITUTION
            self.sendLine('INSTITUTION ' + self.factory.institution)
            return
        elif self.state == INSTITUTION:
            if line != 'OK':
                raise Exception('INSTITUTION: expected OK, got %s' % line)
            self.state = REQUEST
            self.sendLine("REQUEST " + self.factory.request_type)
            for data in self.factory.request_data:
                self.sendLine(data)
            self.sendLine("END")
            return
        elif self.state == REQUEST:
            self.state = STATUS
            self.request_id = line
            self.sendLine("STATUS " + self.request_id)
            return
        elif self.state == STATUS:
            if not 'status="' in line:
                raise Exception('STATUS: expected XML, got %s' % line)
            if 'status="OK"' not in line:
                reactor.callLater(5, self.sendLine,
                                  "STATUS " + self.request_id)
                return
            self.state = DOWNLOAD
            self.sendLine("DOWNLOAD " + self.request_id)
            return
        elif self.state == DOWNLOAD:
            if line == 'ERROR':
                self.state = STATUS
                reactor.callLater(5, self.sendLine,
                                  "STATUS " + self.request_id)
                return
            self.state = DOWNLOAD2
            self.size = int(line)
            self.clock = time.clock()
            self.setRawMode()
            return
        elif self.state == PURGE:
            if line != 'OK':
                raise Exception('PURGE: expected OK got %s' % line)
        self.sendLine("BYE")
        #self.content.close()
        self.transport.loseConnection()

    def rawDataReceived(self, data):
        pos = self.content.tell()
        if pos + len(data) < self.size:
            self.content.write(data)
            if self.factory.debug:
                print '... (data chunk)', pos, 'of', self.size
            return
        self.content.write(data[0:-5])
        self.setLineMode()
        if self.factory.debug:
            print '... (data chunk)', self.content.tell(), 'of', self.size
            print '>>> END', '(%.3lf seconds)' % (time.clock() - self.clock)
        self.state = PURGE
        self.sendLine("PURGE " + self.request_id)
        self.content.close()
        return


class ArcLinkClientFactory(protocol.ClientFactory):
    protocol = ArcLinkClientProtocol

    def __init__(self, user="Anonymous", institution="Anonymous",
                 request_type="WAVEFORM format=MSEED",
                 request_data=["2008,1,1,0,0,0 2008,1,1,1,0,0 BW ROTZ EHE .", ],
                 outfile='temp.mseed', debug=False):
        self.user = user
        self.institution = institution
        self.request_type = request_type
        self.request_data = request_data
        self.outfile = outfile
        self.debug = debug

    def startedConnecting(self, connector):
        print "CONNECTED"

    def clientConnectionFailed(self, connector, reason):
        print 'connection failed:', reason.getErrorMessage()
        reactor.stop()

    def clientConnectionLost(self, connector, reason):
        print 'connection lost:', reason.getErrorMessage()
        reactor.stop()


if __name__ == '__main__':
    factory = ArcLinkClientFactory(request_type="WAVEFORM format=MSEED",
        request_data=["2008,1,1,0,0,0 2008,1,1,1,0,0 BW ROTZ EHE .", ],
        outfile='output.mseed', debug=True)
    reactor.connectTCP('erde.geophysik.uni-muenchen.de', 18001, factory)
    reactor.run()
