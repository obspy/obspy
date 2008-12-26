#!/usr/bin/python

from twisted.internet import protocol
from twisted.protocols import basic
from twisted.protocols import policies
from twisted.internet import reactor
from twisted.internet import defer

import time

OK        = 0
ERROR     = 1
VERSION   = 2
REQUEST   = 3
STATUS    = 4

class ArcLinkClient(basic.LineReceiver, policies.TimeoutMixin):

    # If enabled then log ArcLink communication
    debug = True
    
    # Number of seconds to wait before timing out a connection.  If
    # None, perform no timeout checking.
    timeout = None


    def __init__(self, user="Anonymous", 
                       institute="Anonymous", 
                       requesttype = "INVENTORY",
                       requestdata =["2005,1,1,0,0,0 2005,12,31,0,0,0 GE *",]):
        self.user = user or 'Anonymous'
        self.institute = institute or 'Anonymous'
        self.requesttype = requesttype
        self.requestdata = requestdata


    def sendLine(self, line):
        # Log sendLine only if you are in debug mode for performance
        if self.debug:
            print '>>> ' + line
        basic.LineReceiver.sendLine(self,line)


    def connectionMade(self):
        self.setTimeout(self.timeout)
        self.state_HELLO()

    def lineReceived(self, line):
        self.resetTimeout()

	# Log lineReceived only if you are in debug mode for performance
	if self.debug:
            print '<<< ' + line

        if line=="ERROR":
            return self._failresponse()

        # Catch two lines for version + instution string
        if self._expected==VERSION:
          self._content+=1
          if self._content<2:
            return
        
        if self._expected==STATUS:
            if not 'ready=="true"' in line:
              time.sleep(10)
              return self.state_STATUS()
            else:
              return self._okresponse()
        
        if self._expected==REQUEST:
            self._reqid = line
            return self._okresponse()
        
        if line=="OK":
            return self._okresponse()
        else:
            return self._failresponse()

    def run(self):
        d = state_HELLO()
        d.addCallback(state_USER)
        d.addCallback(state_INSTITUTION)
 

    def state_HELLO(self):
        d = defer.Deferred()
        self.sendLine("HELLO")
        self._content = 0
        self._expected = VERSION
        self._okresponse = self.state_USER
        self._failresponse = self.state_USER
        return d
  
    def state_USER(self):
        self.sendLine("USER "+self.user)
        self._expected = OK
        self._okresponse = self.state_INSTITUTION
        self._failresponse = self.state_BYE


    def state_INSTITUTION(self):
        self.sendLine("INSTITUTION "+self.user)
        self._expected = OK
        self._okresponse = self.state_BYE
        self._failresponse = self.state_BYE


    def state_REQUEST(self):
        self.sendLine("REQUEST "+self.requesttype)
        for r in self.requestdata:
          self.sendLine(r)
        self.sendLine("END")
        self._expected = REQUEST
        self._okresponse = self.state_STATUS
        self._failresponse = self.state_BYE


    def state_STATUS(self):
        self.sendLine("STATUS "+self._reqid)
        self._content = ""
        self._expected = STATUS
        self._okresponse = self.state_PURGE
        self._failresponse = self.state_STATUS


    def state_PURGE(self):
        self.sendLine("PURGE "+self._reqid)
        self._expected = OK
        self._okresponse = self.state_BYE
        self._failresponse = self.state_BYE

  
    def state_BYE(self):
        self.sendLine("BYE")
        self.transport.loseConnection()
        

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