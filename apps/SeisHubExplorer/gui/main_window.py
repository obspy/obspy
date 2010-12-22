# -*- coding: utf-8 -*-

from PyQt4 import QtCore, QtGui

from obspy.core import UTCDateTime

from navigation_bar import NavigationBar
from networks import NetworkTree
from waveforms import Waveforms
from website import Website
from edit_channel_lists_dialog import EditChannelListsDialog
from call_picker_dialog import CallPickerDialog
from picks import Picks
import utils
import os


class MainWindow(QtGui.QWidget):
    """
    This is the main tab of the database viewer.
    """
    def __init__(self, env, parent=None):
        """
        Build the main window.
        """
        QtGui.QWidget.__init__(self, parent)
        self.env = env
        self.env.main_window = self
        # Set title and layout for the main_window.
        self.setWindowTitle('Database Viewer')
        grid = QtGui.QGridLayout()

        # Add a frame to group all plot related gui elements.
        self.plotFrame = QtGui.QGroupBox()
        # XXX: This style sheet results in very flickery graphics...i have no
        # idea why that would happen.
        #self.plotFrame.setStyleSheet('border:1px solid #AAAAAA;')
        self.plotFrameLayout = QtGui.QVBoxLayout()
        self.plotFrameLayout.setMargin(1)
        self.plotFrame.setLayout(self.plotFrameLayout)
        grid.addWidget(self.plotFrame, 0, 0, 12, 8)

        # Add plot status bar. Needs to be done before initializing the
        # Waveform class.
        self.plotStatus = QtGui.QLabel(' ')
        self.plotStatus.setStyleSheet('border:0px;')
        font = QtGui.QFont()
        font.setPointSize(9)

        # Add the waveform viewer.
        self.waveforms = Waveforms(env=self.env)
        self.plotFrameLayout.addWidget(self.waveforms)

        # Add navigation widget.
        self.navigation = NavigationBar(env=self.env)
        #self.plotControlLayout.addWidget(self.navigation)
        self.plotFrameLayout.addWidget(self.navigation)
        self.plotStatus.setFont(font)
        self.plotFrameLayout.addWidget(self.plotStatus)

        # Add the menu for the time_selection.
        times = QtGui.QGroupBox('Timeframe')
        grid.addWidget(times, 0, 8, 1, 2)
        # Time frame layout.
        time_layout = QtGui.QGridLayout()
        # Labels.
        start_label = QtGui.QLabel('Starttime:')
        time_layout.addWidget(start_label, 0, 0)
        end_label = QtGui.QLabel('Endtime:')
        time_layout.addWidget(end_label, 1, 0)
        # Init date selector.
        self.start_date = QtGui.QDateTimeEdit(utils.toQDateTime(self.env.starttime))
        self.start_date.setCalendarPopup(True)
        time_layout.addWidget(self.start_date, 0, 1)
        self.end_date = QtGui.QDateTimeEdit(utils.toQDateTime(self.env.endtime))
        self.end_date.setCalendarPopup(True)
        time_layout.addWidget(self.end_date, 1, 1)
        self.time_button = QtGui.QPushButton('Apply')
        time_layout.addWidget(self.time_button, 2, 1)
        times.setLayout(time_layout)

        QtCore.QObject.connect(self.time_button, QtCore.SIGNAL("clicked()"),
                               self.applyTimes)

        # Add the pick group box.
        self.picks = Picks(env=self.env)
        grid.addWidget(self.picks, 1, 8, 1, 2)
        # Add the network tree.

        self.nw_tree = NetworkTree(self.waveforms, env=self.env)
        grid.addWidget(self.nw_tree, 2, 8, 9, 2)

        # Add the edit lists button.
        self.channelListFrame = QtGui.QFrame()
        self.horizontalChannelListLayout = QtGui.QHBoxLayout()
        self.horizontalChannelListLayout.setMargin(0)
        self.channelListFrame.setLayout(self.horizontalChannelListLayout)
        self.editChannelListsButton = QtGui.QPushButton('Edit Groups')
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding,
                                        QtGui.QSizePolicy.Minimum)
        self.horizontalChannelListLayout.addItem(spacerItem)
        self.horizontalChannelListLayout.addWidget(self.editChannelListsButton)
        grid.addWidget(self.channelListFrame, 11, 8, 1, 2)
        QtCore.QObject.connect(self.editChannelListsButton, QtCore.SIGNAL("clicked()"),
                               self.callEditChannelListDialog)

        # Set the layout and therefore display everything.
        self.setLayout(grid)

    def callEditChannelListDialog(self):
        dialog = EditChannelListsDialog(self.env)
        dialog.exec_()

    def callPicker(self, starttime, endtime):
        """
        Calls the picker.
        """
        # Get the call from the environment.
        command = self.env.picker_command
        # Parse the call.
        command = command.replace('$starttime$',
                      starttime.strftime(self.env.picker_strftime))
        command = command.replace('$endtime$',
                      endtime.strftime(self.env.picker_strftime))
        command = command.replace('$duration$', str(int(endtime - starttime)))
        # Get all channels.
        ids = []
        for waveform in self.waveforms.waveform_scene.waveforms:
            ids.append(waveform.stream[0].id)
        channels = self.env.channel_seperator.join(ids)
        channels = self.env.channel_enclosure[0] + channels + \
                   self.env.channel_enclosure[1]
        command = command.replace('$channels$', channels)
        dialog = CallPickerDialog(self.env, command)
        dialog.exec_()

    def applyTimes(self):
        """
        Changes the times.
        """
        starttime = utils.fromQDateTime(self.start_date.dateTime())
        endtime = utils.fromQDateTime(self.end_date.dateTime())
        self.changeTimes(starttime, endtime)

    def changeTimes(self, new_starttime, new_endtime):
        self.env.st.showMessage('Changing Times...')
        ns = new_starttime
        ne = new_endtime
        new_range = ne - ns
        # Do nothing if the times are the same.
        if self.env.starttime == new_starttime and \
           self.env.endtime == new_endtime:
            return
        # Do not zoom in more than possible to avoid resampling errors.
        if new_range < self.env.maximum_zoom_level:
            half = self.env.maximum_zoom_level / 2.0
            middle = ns + new_range / 2.0
            ns = middle - half
            ne = ns + self.env.maximum_zoom_level
        self.env.starttime = UTCDateTime(ns.year, ns.month, ns.day, ns.hour,
                                         ns.minute)
        self.env.endtime = UTCDateTime(ne.year, ne.month, ne.day, ne.hour,
                                         ne.minute)
        self.env.time_range = self.env.endtime - self.env.starttime
        self.waveforms.redraw()
        self.start_date.setDateTime(utils.toQDateTime(self.env.starttime))
        self.end_date.setDateTime(utils.toQDateTime(self.env.endtime))
        # Set the labels to the new values.
        self.env.st.showMessage('')

    def graphics_start(self):
        self.waveforms.startup()

    def startup(self):
        """
        Some stuff that should get called after everything is loaded.
        """
        self.env.seishub.startup()
        self.nw_tree.startup()

        # Connect some slots.
        QtCore.QObject.connect(self.nw_tree.nw_select_model,
                               QtCore.SIGNAL("selectionChanged(QItemSelection, QItemSelection)"), \
                               self.waveforms.waveform_scene.add_channel)

        web = Website(env=self.env)
        web.startup()
        # Add a WebView to later display the map.
        file = open(os.path.join(self.env.temp_res_dir, 'map.html'))
        html = file.read()
        file.close()
        self.env.web.setHtml(html)
        self.picks.update()

        css_url = QtCore.QUrl.fromLocalFile(os.path.abspath(self.env.css))

        server = '%s/manage/seismology/stations' % self.env.seishub_server
        url = QtCore.QUrl(server)
        url.setUserName(self.env.seishub_user)
        url.setPassword(self.env.seishub_password)
        # Might work with some Qt version...
        self.env.station_browser.page().settings().setUserStyleSheetUrl(css_url)
        self.env.station_browser.load(url)
        self.env.station_browser.page().settings().setUserStyleSheetUrl(css_url)
