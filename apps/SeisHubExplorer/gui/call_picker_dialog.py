#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt4 import QtCore, QtGui

import subprocess

class CallPickerDialog(QtGui.QDialog):
    """
    Writes and reads the xml file with the channel lists.
    """
    def __init__(self, env, command):
        """
        Standart edit.
        """
        QtGui.QDialog.__init__(self)
        #super(CallPickerDialog, self).__init__()
        self.env = env
        self.command = command
        self._initInterface()
        self._connectSignals()
        # Toggle the channel wildcards.
        self._createWildcardCommand()
        self.wildcards_toggled = True
        self.command_edit.setText(self.wildcard_command)

    def _initInterface(self):
        """
        Constructs the interface.
        """
        self.setWindowTitle('Call Picker')
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)
        desc = 'Edit the command for the picker if necessary and press OK to execute it.'
        self.description_label = QtGui.QLabel(desc)
        self.layout.addWidget(self.description_label)
        self.command_edit = QtGui.QLineEdit(self.command)
        self.layout.addWidget(self.command_edit)
        # Frame for the buttons.
        self.button_frame = QtGui.QFrame()
        self.layout.addWidget(self.button_frame)
        self.button_layout = QtGui.QHBoxLayout()
        self.button_layout.setMargin(0)
        self.button_frame.setLayout(self.button_layout)
        self.channel_wildcards_button = \
                QtGui.QPushButton('Toggle Component Wildcards')
        self.button_layout.addWidget(self.channel_wildcards_button)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding,
                                        QtGui.QSizePolicy.Minimum)
        self.button_layout.addItem(spacerItem)
        # Add buttons.
        self.reset_button = QtGui.QPushButton('Reset')
        self.cancel_button = QtGui.QPushButton('Cancel')
        self.ok_button = QtGui.QPushButton('OK')
        self.button_layout.addWidget(self.reset_button)
        self.button_layout.addWidget(self.cancel_button)
        self.button_layout.addWidget(self.ok_button)
        self.ok_button.setDefault(True)

    def _createWildcardCommand(self):
        """
        Creates a wildcard version of the command where all components will be
        replaced by wildcards.
        """
        # Replaces all components with a wildcards and removes duplicate
        # entries.
        channels = self.command.split("'")[1].split(',')
        channels = [str(channel.split()[0])[:-1] + '*' for channel in channels]
        channels = list(set(channels))
        channels.sort()
        # Recreate the new wildcard command.
        self.wildcard_command = self.command.split("'")[0] + "'%s'" %\
            ','.join(channels)

    def toggleChannelWildcards(self):
        """
        Toggles the channel wildcards. True by default.
        """
        if self.wildcards_toggled:
            # Check if changed.
            if str(self.command_edit.text()) != self.wildcard_command:
                msg =  20 * ' ' + 'Command changed!' + 20 * ' '
                info = 'All manual changes will be lost. Continue?'
                box = QtGui.QMessageBox()
                box.setText(msg)
                box.setInformativeText(info)
                box.setStandardButtons(QtGui.QMessageBox.No| QtGui.QMessageBox.Yes)
                ret_code = box.exec_()
                if ret_code == QtGui.QMessageBox.No:
                    return
            self.command_edit.setText(self.command)
            self.wildcards_toggled = False
        else:
            # Check if changed.
            if str(self.command_edit.text()) != self.command:
                msg =  20 * ' ' + 'Command changed!' + 20 * ' '
                info = 'All manual changes will be lost. Continue?'
                box = QtGui.QMessageBox()
                box.setText(msg)
                box.setInformativeText(info)
                box.setStandardButtons(QtGui.QMessageBox.No| QtGui.QMessageBox.Yes)
                ret_code = box.exec_()
                if ret_code == QtGui.QMessageBox.No:
                    return
            self.command_edit.setText(self.wildcard_command)
            self.wildcards_toggled = True

    def reset(self):
        """
        Resets the content of the command edit line.
        """
        if self.wildcards_toggled:
            self.command_edit.setText(self.wildcard_command)
        else:
            self.command_edit.setText(self.command)

    def accept(self):
        """
        Executes the command.
        """
        command = str(self.command_edit.text())
        subprocess.Popen(command, shell=True)
        QtGui.QDialog.accept(self)
        #super(CallPickerDialog, self).accept()

    def _connectSignals(self):
        """
        Connects signals and slots.
        """
        QtCore.QObject.connect(self.reset_button, QtCore.SIGNAL("clicked()"),
                               self.reset)
        QtCore.QObject.connect(self.cancel_button, QtCore.SIGNAL("clicked()"),
                               self.reject)
        QtCore.QObject.connect(self.ok_button, QtCore.SIGNAL("clicked()"),
                               self.accept)
        QtCore.QObject.connect(self.channel_wildcards_button, QtCore.SIGNAL("clicked()"),
                               self.toggleChannelWildcards)
