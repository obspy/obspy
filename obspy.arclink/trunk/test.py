#!/usr/bin/python

from twisted.internet import defer, protocol, reactor
from twisted.protocols import basic, policies
import time
from StringIO import StringIO


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


class ArcLinkClient(basic.LineReceiver, policies.TimeoutMixin):

    # If enabled then log ArcLink communication
    debug = True

    # Number of seconds to wait before timing out a connection.  If
    # None, perform no timeout checking.
    timeout = None
    MAX_LENGTH = 1024 * 1024

    def __init__(self, user="Anonymous",
                       institute="Anonymous",
                       requesttype="WAVEFORM",
                       requestdata=["2008,1,1,0,0,0 2008,1,1,1,0,0 BW ROTZ EHE .", ],
                       outfile='temp.mseed'):
        self.user = user or 'Anonymous'
        self.institute = institute or 'Anonymous'
        self.requesttype = requesttype
        self.requestdata = requestdata
        self.outfile = outfile


    def sendLine(self, line):
        # Log sendLine only if you are in debug mode for performance
        if self.debug:
            print '>>> ' + line
        basic.LineReceiver.sendLine(self, line)


    def connectionMade(self):
        self.content = StringIO()
        self.setTimeout(self.timeout)
        self.state = HELLO
        self.sendLine('HELLO')

    def lineReceived(self, line):
        if self.debug:
            print '<<< ' + line[0:10], self.state

        if self.state == HELLO:
            self.state = HELLO2
            return
        elif self.state == HELLO2:
            self.state = USER
            self.sendLine('USER ' + self.user)
            return
        elif self.state == USER:
            if line != 'OK':
                raise
            self.state = INSTITUTION
            self.sendLine('INSTITUTION ' + self.user)
            return
        elif self.state == INSTITUTION:
            if line != 'OK':
                raise
            self.state = REQUEST
            self.sendLine("REQUEST " + self.requesttype)
            for r in self.requestdata:
                self.sendLine(r)
            self.sendLine("END")
            return
        elif self.state == REQUEST:
            self.state = STATUS
            self.request_id = line
            self.sendLine("STATUS " + self.request_id)
            return
        elif self.state == STATUS:
            if not 'status="' in line:
                raise
            if 'status="OK"' not in line:
                reactor.callLater(5, self.sendLine, "STATUS " + self.request_id)
                return
            self.state = DOWNLOAD
            self.sendLine("DOWNLOAD " + self.request_id)
            return
        elif self.state == DOWNLOAD:
            if line == 'ERROR':
                self.state = STATUS
                reactor.callLater(5, self.sendLine, "STATUS " + self.request_id)
                return
            self.state = DOWNLOAD2
            self.size = int(line)
            if self.size > self.MAX_LENGTH:
                raise Exception("MaxLength!!")
            self.setRawMode()
            return
        elif self.state == PURGE:
            if line != 'OK':
                raise
            self.state = BYE
            return
        self.sendLine("BYE")
        self.content.close()
        self.transport.loseConnection()

    def rawDataReceived(self, data):
        if self.content.pos + len(data) < self.size:
            self.content.write(data)
            if self.debug:
                print '...', self.content.pos, 'of', self.size
            return
        self.content.write(data[0:-5])
        self.setLineMode()
        self.state = PURGE
        self.sendLine("PURGE " + self.request_id)
        if self.debug:
            print '...', self.content.pos, 'of', self.size
        fh = open(self.outfile, 'w')
        self.content.seek(0)
        fh.write(self.content.read())
        fh.close()
        return



class ArcLinkClientFactory(protocol.ClientFactory):
    protocol = ArcLinkClient

    def startedConnecting(self, connector):
        print "CONNECTED"

    def clientConnectionFailed(self, connector, reason):
        print 'connection failed:', reason.getErrorMessage()
        reactor.stop()

    def clientConnectionLost(self, connector, reason):
        print 'connection lost:', reason.getErrorMessage()
        reactor.stop()


if __name__ == '__main__':
    factory = ArcLinkClientFactory()
    reactor.connectTCP('erde.geophysik.uni-muenchen.de', 18001, factory)
    reactor.run()
