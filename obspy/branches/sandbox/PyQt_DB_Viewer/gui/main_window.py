from PyQt4 import QtCore, QtGui, QtWebKit

from networks import NetworkTree
from waveforms import Waveforms
from website import Website
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
        # Set title and layout for the main_window.
        self.setWindowTitle('Database Viewer')
        grid = QtGui.QGridLayout()

        # Add the waveform viewer.
        self.waveforms = Waveforms(env = self.env)
        grid.addWidget(self.waveforms, 0, 0, 12, 7)

        # Add the menu for the time_selection.
        times = QtGui.QGroupBox('Timeframe')
        grid.addWidget(times, 0,8,1,2)
        # Time frame layout.
        time_layout = QtGui.QGridLayout()
        # Labels.
        start_label = QtGui.QLabel('Starttime:')
        time_layout.addWidget(start_label,0,0)
        end_label = QtGui.QLabel('Endtime:')
        time_layout.addWidget(end_label,1,0)
        # Init date selector.
        self.start_date = QtGui.QDateTimeEdit(utils.toQDateTime(self.env.starttime))
        self.start_date.setCalendarPopup(True)
        time_layout.addWidget(self.start_date,0,1)
        self.end_date = QtGui.QDateTimeEdit(utils.toQDateTime(self.env.endtime))
        self.end_date.setCalendarPopup(True)
        time_layout.addWidget(self.end_date,1,1)
        self.time_button = QtGui.QPushButton('Apply')
        time_layout.addWidget(self.time_button, 2, 1)
        times.setLayout(time_layout)

        QtCore.QObject.connect(self.time_button, QtCore.SIGNAL("clicked()"),
                               self.changeTimes)

        # Add the pick group box.
        self.picks = Picks(env = self.env)
        grid.addWidget(self.picks, 1, 8, 1, 2)
        # Add the network tree.

        self.nw_tree = NetworkTree(self.waveforms, env = self.env)
        grid.addWidget(self.nw_tree, 2, 8, 10, 2)

        # Set the layout and therefore display everything.
        self.setLayout(grid)

    def changeTimes(self):
        """
        Changes the times.
        """
        starttime = utils.fromQDateTime(self.start_date.dateTime())
        endtime = utils.fromQDateTime(self.end_date.dateTime())
        # Do nothing if the times are the same.
        if self.env.starttime == starttime and self.env.endtime == endtime:
            return
        self.env.starttime = starttime
        self.env.endtime = endtime
        self.env.time_range = endtime - starttime
        self.waveforms.scene.redraw()

    def graphics_start(self):
        self.waveforms.scene.startup()

    def startup(self):
        """
        Some stuff that should get called after everything is loaded.
        """
        self.env.seishub.startup()
        self.nw_tree.startup()

        # Connect some slots.
        # XXX: New method not working with PyQt4
        # self.nw_tree.nw_select_model.selectionChanged.connect(self.waveforms.scene.add_channel)
        #QtCore.SLOT("self.waveforms.scene.add_channel(int, int)")
        QtCore.QObject.connect(self.nw_tree.nw_select_model, QtCore.SIGNAL("selectionChanged(QItemSelection, QItemSelection)"),\
			       self.waveforms.scene.add_channel)

        web = Website(env = self.env)
        # Add a WebView to later display the map.
        file = open(os.path.join(self.env.res_dir, 'map.html'))
        html = file.read()
        file.close()
        self.env.web.setHtml(html)
        self.picks.update()

        css_url = QtCore.QUrl.fromLocalFile(os.path.abspath(self.env.css))

        server = '%s/manage/seismology/stations' % self.env.seishub_server
        print server
        url = QtCore.QUrl(server)
        url.setUserName('admin')
        url.setPassword('admin')
        # Might work with some Qt version...
        self.env.station_browser.page().settings().setUserStyleSheetUrl(css_url)
        self.env.station_browser.load(url)
        self.env.station_browser.page().settings().setUserStyleSheetUrl(css_url)
