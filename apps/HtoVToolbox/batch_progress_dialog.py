# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: batch_process_dialog.py
#  Purpose: Provides a dialogue box to perform batch HVSR.
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#  License: GPLv2
#
# Copyright (C) 2010 Lion Krischer
#---------------------------------------------------------------------

from PyQt4 import QtCore, QtGui

from batch_progress import Ui_BatchDialog


class BatchProgressDialog(QtGui.QDialog):
    """
    Main Window with the design loaded from the Qt Designer.
    """
    def __init__(self):
        """
        Standard init.
        """
        QtGui.QDialog.__init__(self)
        self.cancelPressed = False
        # This is how the init is called in the file created py pyuic4.
        self.design = Ui_BatchDialog()
        self.design.setupUi(self)
        self.major_value=0
        self.__connectSignals()

    def setMinimum(self, min):
        """
        Sets the minimum of the progress bar.
        """
        self.design.progressBar.setMinimum(100*min)

    def setMaximum(self, max):
        """
        Sets the maximum of the progress bar.
        """
        self.design.progressBar.setMaximum(100*max)

    def setRange(self, min, max):
        """
        Sets the range of the progress bar.
        """
        self.design.progressBar.setRange(100*min, 100*max)

    def setValue(self, value):
        """
        Sets the value of the progress bar.
        """
        self.major_value=100*value
        self.design.progressBar.setValue(100*value)

    def setMinorFracValue(self, value, count):
        """
        Sets the minor value to value/count percent inbetween two major values.
        """
        value = self.major_value + float(value)/count*100
        self.design.progressBar.setValue(value)

    def minimum(self):
        return self.design.progressBar.minimum()/100

    def maximum(self):
        return self.design.progressBar.minimum()/100

    def value(self):
        return self.design.progressBar.value()/100

    def setMajorLabel(self, text):
        """
        Sets the major Label.
        """
        self.design.majorLabel.setText(text)

    def setMinorLabel(self, text):
        """
        Sets the minor Label.
        """
        self.design.minorLabel.setText(text)

    def setStatusLabel(self, text):
        """
        Sets the status Label.
        """
        self.design.statusLabel.setText(text)

    def cancelButtonPressed(self):
        """
        Variable to keep track of the cancel button.
        """
        self.setMajorLabel('Cancelling...')
        self.cancelPressed = True
        # Emit signal.
        self.emit(QtCore.SIGNAL('cancelButtonPressed()'))

    def __connectSignals(self):
        """
        Connects the signals.
        """
        QtCore.QObject.connect(self.design.cancelButton,
                               QtCore.SIGNAL('clicked()'),
                               self.cancelButtonPressed)
