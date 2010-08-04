from PyQt4 import QtCore, QtGui

class Picks(QtGui.QGroupBox):
    def __init__(self, env, parent = None, *args, **kwargs):
        QtGui.QGroupBox.__init__(self, 'Picks')
        #super(Picks, self).__init__('Picks')
        self.env = env

        self.pick_layout = QtGui.QVBoxLayout()


        # Combo box chooser.
        self.combo_box = QtGui.QComboBox()
        self.pick_layout.addWidget(self.combo_box)
        self.combo_box.addItem('Overview')
        self.combo_box.addItem('Detail')


        self.overviewFrame = QtGui.QFrame()
        self.pick_layout.addWidget(self.overviewFrame)
        self.detailFrame = QtGui.QFrame()
        self.pick_layout.addWidget(self.detailFrame)

        self.overviewLayout = QtGui.QVBoxLayout()
        self.detailLayout = QtGui.QVBoxLayout()

        # Get events and picks from the database for the current timeframe.
        event_count, pick_count = self.env.db.getPickAndEventCount()
        # stats.
        msg = '%i events with %i picks for the\nchosen timeframe.\n' %\
            (event_count, pick_count)
        self.pick_overview = QtGui.QLabel(msg)
        self.pick_overview.setWordWrap(True)
        self.overviewLayout.addWidget(self.pick_overview)

        # Connect the signal emmited when the combo box is changed.
	# XXX: Not working with PyQt 4.4
        # self.combo_box.currentIndexChanged.connect(self.indexChanged)
	# Create slot.
	#QtCore.SLOT("self.update(int)")
	QtCore.QObject.connect(self.combo_box, QtCore.SIGNAL("currentIndexChanged(int)"),\
			       self.indexChanged)

    def update(self):
        # Get events and picks from the database for the current timeframe.
        event_count, pick_count = self.env.db.getPickAndEventCount()
        # stats.
        msg = '%i events with %i picks for the\nchosen timeframe.\n' %\
            (event_count, pick_count)
        self.pick_overview.setText(msg)
        # channels.
        channel_ids = self.env.db.getChannelsWithPicks()
        self.detail_label = QtGui.QLabel('Channels with picks:')
        self.detailLayout.addWidget(self.detail_label)
        self.pick_channels = QtGui.QTableWidget(len(channel_ids), 1)
        self.pick_channels.setAlternatingRowColors(True)
        # Hide the headers.
        self.pick_channels.verticalHeader().hide()
        self.pick_channels.horizontalHeader().hide()
        # Stretch the one row.
        self.pick_channels.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeMode(1))
        # Add every channel with picks.
        for _i, channel in enumerate(channel_ids):
            item = QtGui.QTableWidgetItem(channel)
            item.setFlags(QtCore.Qt.ItemFlag(1))
            #item.setFlags(QtCore.Qt.ItemIsSelectable)
            self.pick_channels.setItem(_i,0, item)
        self.detailLayout.addWidget(self.pick_channels)

        self.overviewFrame.setLayout(self.overviewLayout)
        self.detailFrame.setLayout(self.detailLayout)

        self.detailFrame.hide()

        self.setLayout(self.pick_layout)

    # The decorator is necessary because the signal emitted is overloaded and
    # just the integer one is handled here.
    # XXX: The decorator needs at least PyQt 4.5
    # @QtCore.pyqtSlot(int)
    def indexChanged(self, index):
	if type(index) != int:
	    return
        if index == 0:
            self.overviewFrame.show()
            self.detailFrame.hide()
        elif index == 1:
            self.overviewFrame.hide()
            self.detailFrame.show()

