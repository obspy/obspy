# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: overlay.py
#  Purpose: Provides a waiting indicator overlay.
#           Adapted from  http://www.diotavelli.net/PyQtWiki/A full
#           widget waiting indicator
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#  License: GPLv2
#
# Copyright (C) 2010 Lion Krischer
#---------------------------------------------------------------------

import math
from PyQt4.QtCore import Qt, QThread
from PyQt4 import QtGui, QtCore

class Overlay(QtGui.QWidget):
    """
    Add overlay to show it is busy.
    """
    def __init__(self, parent, hv):
        QtGui.QWidget.__init__(self, parent)
        self.hv = hv
        palette = QtGui.QPalette(self.palette())
        palette.setColor(palette.Background, Qt.transparent)
        self.setPalette(palette)

    def paintEvent(self, event):
        """
        Paint it.
        """
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(event.rect(), QtGui.QBrush(QtGui.QColor(255, 255, 255, 127)))
        painter.setPen(QtGui.QPen(Qt.NoPen))

        for i in range(6):
            if (self.counter / 5) % 6 == i:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(127 + (self.counter % 5)*32, 127, 127)))
            else:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(127, 127, 127)))
            painter.drawEllipse(
                self.width()/2 + 30 * math.cos(2 * math.pi * i / 6.0) - 10,
                self.height()/2 + 30 * math.sin(2 * math.pi * i / 6.0) - 10,
                20, 20)
        painter.end()

    def showEvent(self, event):
        """
        Overwrite the show event.
        """
        self.counter = 0
        self.timer = OverlayTimer(self)

class OverlayTimer(QThread):
    def __init__(self, overlay):
        super(OverlayTimer, self).__init__()
        self.overlay = overlay
        self.start()

    def updateOverlay(self):
        self.overlay.counter += 1
        self.overlay.update()

    def run(self):
        self.overlay_timer = QtCore.QTimer()
        # Connect the timer to update the things.
        QtCore.QObject.connect(self.overlay_timer, QtCore.SIGNAL("timeout()"),\
                       self.updateOverlay) 
        # Call every 50 msecs.
        self.overlay_timer.start(50)
        # Start main loop of the thread. Needed for the timers.
        self.exec_()
