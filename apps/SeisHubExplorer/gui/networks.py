# -*- coding: utf-8 -*-

from PyQt4 import QtCore, QtGui
import pickle


class TreeItem(object):
    """
    Simple Tree item.
    """
    def __init__(self, data, what, parent=None):
        self.parentItem = parent
        self.itemData = data
        self.type = what
        self.childItems = []
        self.createFullPath()
        # Default value for group. Will only be True if it is a channel group.
        self.channel_group = False

    def appendChild(self, item):
        self.childItems.append(item)

    def child(self, row):
        return self.childItems[row]

    def childCount(self):
        return len(self.childItems)

    def columnCount(self):
        return 2

    def data(self, column):
        if type(self.itemData) == tuple and column == 0:
            return self.itemData[0]
        if type(self.itemData) == tuple and column == 1:
            data = self.itemData[1]
            if type(data) == str:
                return data
            return data['station_name']
        elif column == 0:
            return self.itemData
        else:
            return ''

    def tooltip(self):
        msg = '%s %s' % (self.type.capitalize(), self.getFullPath()[0])
        data = self.itemData
        if type(data) == tuple and type(data[1]) == dict:
            data = data[1]
            msg += '\n%s\nLat: %s\nLong: %s\nEle: %s' \
                    % (data['station_name'], data['latitude'],
                    data['longitude'], data['elevation'])
        return msg


    def getFullPath(self):
        """
        Custom method to return the full path and the type of the item.

        Returns a tuple with two strings, e.g.
            ('BW.ALTM..EHE', 'station') or
            ('BW', 'network')
        """
        return self.fullPath

    def createFullPath(self):
        """
        Creates the full path.
        """
        if self.type == 'network':
            path = self.data(0)
        elif self.type == 'station':
            path = '%s.%s' % (self.parentItem.data(0), self.data(0))
        elif self.type == 'channel':
            data = self.data(0)
            # No seperate location handling. Is included in the channel data.
            if '.' in data:
                path = '%s.%s.%s' % (self.parentItem.parentItem.data(0),
                                     self.parentItem.data(0), data)
            else:
                path = '%s.%s..%s' % (self.parentItem.parentItem.data(0),
                                     self.parentItem.data(0), data)
        elif self.type == 'channel_list':
            path = self.data(0)
        else:
            # Hopefully does not happen.
            path = 'None'
        self.fullPath = (path, self.type)

    def parent(self):
        return self.parentItem

    def row(self):
        if self.parentItem:
            return self.parentItem.childItems.index(self)

        return 0

class TreeModel(QtCore.QAbstractItemModel):
    """
    Implements the Tree Model.
    """
    def __init__(self, filename, env, parent=None):
        """
        Init Method.
        """
        QtCore.QAbstractItemModel.__init__(self, parent)
        #super(TreeModel, self).__init__(parent)
        self.env = env
        # Lade Datensatz 
        f = open(filename, 'rb')
        try:
            self.networks = pickle.load(f)
            self.server = self.networks.pop('Server')
            self.query_date = self.networks.pop('Date')
        finally:
            f.close()
        # Get all possible channels in one large list.
        self.env.all_channels = []
        nws = self.networks.keys()
        for nw in nws:
            sts = self.networks[nw].keys()
            for st in sts:
                locs = self.networks[nw][st].keys()
                for loc in locs:
                    if loc == 'info':
                        continue
                    chans = self.networks[nw][st][loc][0]
                    for chan in chans:
                        self.env.all_channels.append('%s.%s.%s.%s' % \
                            (nw, st, loc, chan))
        # Set root item.
        self.rootItem = TreeItem(('Networks', 'Details'), 'root')
        self.setupModelData(self.rootItem)

    def columnCount(self, parent):
        if parent.isValid():
            return parent.internalPointer().columnCount()
        else:
            return self.rootItem.columnCount()

    def data(self, index, role):
        item = index.internalPointer()

        if not index.isValid():
            return QtCore.QVariant()

        if role == QtCore.Qt.ToolTipRole:
            return QtCore.QVariant(item.tooltip())

        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        return QtCore.QVariant(item.data(index.column()))

    def getFullPath(self, index):
        """
        Returns the full path of the selection.
        """
        item = index.internalPointer()
        return item.getFullPath()

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.NoItemFlags

        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def headerData(self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self.rootItem.data(section))

        return QtCore.QVariant(None)

    def index(self, row, column, parent):
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()

        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        childItem = parentItem.child(row)
        if childItem:
            return self.createIndex(row, column, childItem)
        else:
            return QtCore.QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QtCore.QModelIndex()

        childItem = index.internalPointer()
        parentItem = childItem.parent()

        if parentItem == self.rootItem:
            return QtCore.QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        return parentItem.childCount()

    def setupModelData(self, parent):
        """
        Reads the dictionary and fills the Tree.
        """
        networks = self.networks.keys()
        networks.sort()
        # Loop over all networks.
        for network in networks:
            # Append network to the root item.
            parent.appendChild(TreeItem((network), 'network', parent))
            # Get the child as shortcut.
            nw_child = parent.child(parent.childCount() - 1)
            nw_dict = self.networks[network]
            stations = nw_dict.keys()
            stations.sort()
            # Loop over all stations.
            for station in stations:
                # Append station.
                info = nw_dict[station]['info']
                if len(info):
                    nw_child.appendChild(TreeItem((station,
                               nw_dict[station]['info']),
                              'station', nw_child))
                else:
                    nw_child.appendChild(TreeItem((station), 'station', nw_child))
                #New cur_child.
                st_child = nw_child.child(nw_child.childCount() - 1)
                st_dict = nw_dict[station]
                locations = st_dict.keys()
                locations.sort()
                # Loop over all locations.
                for location in locations:
                    # Dismiss infos.
                    if location == 'info':
                        continue
                    # Do not append location but rather directly get the
                    # channels.
                    channels = st_dict[location][0]
                    for channel in channels:
                        if len(location):
                            st_child.appendChild(TreeItem((location + '.'\
                                        + channel), 'channel', st_child))
                        else:
                            st_child.appendChild(TreeItem((channel), 'channel',
                                                                     st_child))
                        # Connect to signal.
                        ch_signal = st_child.child(st_child.childCount() - 1)
        # Also fill it with the channel lists.
        parent.appendChild(TreeItem(('Groups'), 'Plot serveral channels.', parent))
        chan_lists = parent.child(parent.childCount() - 1)
        # Loop over every list.
        lists = self.env.channel_lists.keys()
        lists.sort()
        for item in lists:
            tree_item = TreeItem((item), 'channel_list', chan_lists)
            tree_item.channel_group = True
            chan_lists.appendChild(tree_item)

class TreeSelector(QtGui.QItemSelectionModel):
    def __init__(self, model, *args, **kwargs):
        # XXX: Kwargs not working.
        # super(TreeSelector, self).__init__(*args, **kwargs)
        # super(TreeSelector, self).__init__(model)
        QtGui.QItemSelectionModel.__init__(self, model)

class NetworkTree(QtGui.QTreeView):
    def __init__(self, waveforms, env, parent=None, *args, **kwargs):
        QtGui.QTreeView.__init__(self, parent)
        #super(NetworkTree, self).__init__(parent)
        self.env = env
        # Animate the Network tree.
        self.setAnimated(True)
        # Supposedly improves the performance of the TreeView.
        self.setUniformRowHeights(True)

    def startup(self):
        # Init the Network selector.
        nw_model = TreeModel(self.env.network_index, self.env)
        self.setModel(nw_model)
        self.nw_select_model = TreeSelector(nw_model)
        self.setSelectionModel(self.nw_select_model)
        self.setColumnWidth(0, 160)

    def refreshModel(self):
        """
        Creates a new tree model that will contain any changes.
        """
        self.startup()
        # Reconnect the signals.
        QtCore.QObject.connect(self.nw_select_model,
            QtCore.SIGNAL("selectionChanged(QItemSelection, QItemSelection)"), \
            self.env.main_window.waveforms.waveform_scene.add_channel)
