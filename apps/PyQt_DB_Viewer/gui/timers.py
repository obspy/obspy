from PyQt4 import QtCore
from obspy.core import UTCDateTime

class Timers(QtCore.QThread):
    def __init__(self, env, parent = None, *args, **kwargs):
        super(Timers, self).__init__(parent)
        self.env = env
        # Set to lowest priority.
        #self.setPriority(1)
        self.start()

    def updateServerStatus(self):
        msg = 'SeisHub server %s: ' % self.env.seishub_server
        self.env.seishub.ping()
        if self.env.seishub.online:
            msg += '<font color="#339966">connected</font>'
        else:  
            msg += '<font color="#FF0000">no connection</font>'
        self.env.server_status.setText(msg)

    def updateCurrentTime(self):
        msg = 'Current UTC time: '
        cur = UTCDateTime()
        msg += str(cur)[:19]
        self.env.current_time.setText(msg)

    def run(self):
        # Run once.
        self.updateCurrentTime()
        self.updateServerStatus()
        # Setup server timer.
        self.server_timer = QtCore.QTimer()

        # XXX: New method not working with PyQt4
        # self.server_timer.timeout.connect(self.updateServerStatus)
        #QtCore.SLOT("self.updateServerStatus()")
        QtCore.QObject.connect(self.server_timer, QtCore.SIGNAL("timeout()"),\
                       self.updateServerStatus) 
        # Call every ten second.
        self.server_timer.start(10000)
        # Setup time timer.
        self.time_timer = QtCore.QTimer()

        # XXX: New method not working with PyQt4
        # self.time_timer.timeout.connect(self.updateCurrentTime)
        #QtCore.SLOT("self.updateCurrentTime()")
        QtCore.QObject.connect(self.time_timer, QtCore.SIGNAL("timeout()"),\
                       self.updateCurrentTime)

        # Call every  second.
        self.time_timer.start(1000)
        # Start main loop of the thread. Needed for the timers.
        self.exec_()
