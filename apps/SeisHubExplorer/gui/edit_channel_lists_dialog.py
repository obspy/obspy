# -*- coding: utf-8 -*-

from PyQt4 import QtCore, QtGui
from StringIO import StringIO


class EditChannelListsDialog(QtGui.QDialog):
    """
    Writes and reads the xml file with the channel lists.
    """
    def __init__(self, env):
        """
        Standart edit.
        """
        QtGui.QDialog.__init__(self)
        #super(EditChannelListsDialog, self).__init__()
        self.env = env
        self._initInterface()
        self._connectSignals()
        self.fillWithCurrentLists()

    def _initInterface(self):
        """
        Constructs the interface.
        """
        # Set two colors.
        self.red_color = QtGui.QColor(255, 0, 0, 100)
        self.green_color = QtGui.QColor(0, 255, 0, 100)
        self.setWindowTitle('Edit Channel Groups')
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)
        desc = 'There must be exactly one list per line with the following syntax: "LIST_NAME: CHANNEL1; CHANNEL2; ..."'
        self.description_label = QtGui.QLabel(desc)
        self.layout.addWidget(self.description_label)
        desc2 = 'Supports Unix wildcards (?, * and sequences []), e.g. BW.FURT..[E,S]*'
        self.description_label2 = QtGui.QLabel(desc2)
        self.layout.addWidget(self.description_label2)
        self.edit_area = QtGui.QTextEdit()
        self.layout.addWidget(self.edit_area)
        # Frame for the buttons.
        self.button_frame = QtGui.QFrame()
        self.layout.addWidget(self.button_frame)
        self.button_layout = QtGui.QHBoxLayout()
        self.button_layout.setMargin(0)
        self.button_frame.setLayout(self.button_layout)
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

    def fillWithCurrentLists(self):
        """
        Fills the text area with the available list in the right syntax.
        """
        msg = ''
        lists = self.env.channel_lists.keys()
        lists.sort()
        for item in lists:
            msg += '%s: %s\n' % (item, '; '.join(self.env.channel_lists[item]))
        self.edit_area.setText(msg)

    def accept(self):
        """
        Parses whatever is in the the text area and fills the environment
        channel list dictionary.
        """
        stringio = StringIO(str(self.edit_area.toPlainText()).strip())
        channel_dict = {}
        error_occured = False
        self.edit_area.clear()
        for line in stringio:
            ret_val = self._parseLine(line)
            if ret_val is None:
                continue
            elif ret_val is False:
                error_occured = True
                self.edit_area.setTextBackgroundColor(self.red_color)
                self.edit_area.insertPlainText(line)
                continue
            self.edit_area.setTextBackgroundColor(self.green_color)
            self.edit_area.insertPlainText(line)
            key, value = ret_val
            _i = 1
            # Restrict while loop to 100 for savety reasons.
            while _i < 100:
                if _i == 1:
                    cur_name = key
                else:
                    cur_name = '%s_%i' % (key, _i)
                # Check if the name is alreadt in the dictionary, otherwise
                # increment the number.
                if key in channel_dict:
                    _i += 1
                    continue
                channel_dict[key] = value
                break
        self.env.channel_lists = channel_dict
        if error_occured:
            return
        self.env.channel_lists_parser.writeFile()
        self.env.main_window.nw_tree.refreshModel()
        QtGui.QDialog.accept(self)
        #super(EditChannelListsDialog, self).accept()

    def _parseLine(self, line):
        """
        Tries to parse a single line. Will return either a dictionary key and a
        list or False if it fails to parse the line. An empty line will return
        None.
        """
        line = line.strip()
        if not line:
            return None
        items = line.split(':')
        if len(items) != 2:
            return False
        key = items[0]
        values = items[1].strip().split(';')
        values = [item.strip() for item in values if len(item.strip())]
        if len(values) < 1:
            return False
        return key, values

    def _connectSignals(self):
        """
        Connects signals and slots.
        """
        QtCore.QObject.connect(self.reset_button, QtCore.SIGNAL("clicked()"),
                               self.fillWithCurrentLists)
        QtCore.QObject.connect(self.cancel_button, QtCore.SIGNAL("clicked()"),
                               self.reject)
        QtCore.QObject.connect(self.ok_button, QtCore.SIGNAL("clicked()"),
                               self.accept)
