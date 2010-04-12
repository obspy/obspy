from PyQt4 import QtCore, QtGui, QtWebKit

from networks import NetworkTree
from waveforms import Waveforms
from website import Website
from picks import Picks
import utils
import os

class MainWindow(QtGui.QWidget):
    """
    Sets the basic grid layout.
    """
    def __init__(self, env, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.env = env

        self.setWindowTitle('Database Viewer')
        grid = QtGui.QGridLayout()


        self.waveforms = Waveforms(env = self.env)
        #grid.addWidget(QtGui.QTextEdit('Left Side with some dummy text editing.'), 0, 0, 8, 7)
        grid.addWidget(self.waveforms, 0, 0, 12, 7)


        # Add group box.
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
        start_date = QtGui.QDateTimeEdit(utils.toQDateTime(self.env.starttime))
        start_date.setCalendarPopup(True)
        time_layout.addWidget(start_date,0,1)
        end_date = QtGui.QDateTimeEdit(utils.toQDateTime(self.env.endtime))
        end_date.setCalendarPopup(True)
        time_layout.addWidget(end_date,1,1)
        times.setLayout(time_layout)

        # Instance picks.
        self.picks = Picks(env = self.env)
        grid.addWidget(self.picks, 1, 8, 1, 2)

        # Network tree.
        self.nw_tree = NetworkTree(self.waveforms, env = self.env)
        grid.addWidget(self.nw_tree, 2, 8, 10, 2)

    
        # Init the grid.
        self.setLayout(grid)

    def startup(self):
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
