# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: filenamewidget.py
#  Purpose: Provides a dialog to select files for batch operations.
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#  License: GPLv2
#
# Copyright (C) 2010 Lion Krischer
#---------------------------------------------------------------------

from PyQt4 import QtCore, QtGui

import os


class FilenameItem(QtGui.QTreeWidgetItem):
    """
    Custom version of QTreeWidgetItem.
    """
    def __init__(self, string_list, toplevel=True):
        QtGui.QTreeWidgetItem.__init__(self, string_list,
                                       type=QtGui.QTreeWidgetItem.Type)
        if toplevel:
            self.setFlags(QtCore.Qt.ItemIsSelectable |
                QtCore.Qt.ItemIsDropEnabled |
                QtCore.Qt.ItemIsEnabled)
        else:
            self.setFlags(QtCore.Qt.ItemIsSelectable |
                QtCore.Qt.ItemIsDragEnabled |
                QtCore.Qt.ItemIsEnabled)


class FilenameWidget(QtGui.QTreeWidget):
    """
    Handles the filenames.
    """
    def __init__(self, parent):
        """
        Very basic init method.
        """
        QtGui.QTreeWidget.__init__(self, parent)
        # List to keep track of all items.
        self.plain_list = [] 
        # Set the column count.
        self.setColumnCount(2)
        # Enable only internal drag and drop.
        self.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        # Set the header.
        self.setHeaderLabels(['Name/File', 'Size'])
        # Connect signals.
        self._connectSignals()

    def addFilenames(self, filenames):
        """
        Add a list of filenames if they do not already exists.

        :param filenames: list
            list with filenames
        """
        # Keep seperate_list to easily check for duplicates.
        # Remove those already existing.
        checked_filenames = []
        for file in filenames:
            if file in self.plain_list:
                continue
            checked_filenames.append(file)
        # Add to the checklist.
        self.plain_list.extend(checked_filenames)
        new_items = len(checked_filenames)
        # Loop and add the data to the model.
        for file in checked_filenames:
            size = os.path.getsize(file)
            string_list = ['%s' % os.path.basename(file), '']
            #'[1 file - %.2f MB]' % (size/1000000.0)]
            item = FilenameItem(string_list)
            sub_item = FilenameItem([file, '%.2f MB' % (size/1000000.0)],
                                    toplevel=False)
            item.addChild(sub_item)
            self.addTopLevelItem(item)
        # Can't figure out the signal that gets emitted once a new Item is
        # added. So it is done manually.
        self.resizeColumnToContents(0)

    def removeItems(self, items):
        """
        Items is a list with items which will all be removed.
        """
        for item in items:
            # Will be -1 if it is no top level item.
            index = self.indexOfTopLevelItem(item)
            # Differentiate child ...
            if index == -1:
                parent = item.parent()
                child_index = parent.indexOfChild(item)
                # Remove the child.
                child = parent.takeChild(child_index)
                # Remove the name from the list.
                self.plain_list.remove(str(child.text(0)))
                # Figure out if the parent now has some children left and
                # otherwise remove it.
                if not parent.childCount():
                    self.takeTopLevelItem(self.indexOfTopLevelItem(parent))
                else:
                    # Set a new text for the parent items count and size.
                    count = parent.childCount()
                    size = 0
                    for _i in xrange(count):
                        size += os.path.getsize(str(parent.child(_i).text(0)))
                    multiple = '' if count == 1 else 's'
                    new_string = '[%i file%s - %.2f MB]' %\
                        (count, multiple, size/1000000.0)
                    parent.setText(1, new_string)
            # ... and parent items.
            else:
                # Remove all children.
                children = list(item.takeChildren())
                # Remove their names from the filename list.
                for child in children:
                    self.plain_list.remove(str(child.text(0)))
                # Remove the item.
                self.takeTopLevelItem(index)

    def autoMatchFiles(self):
        """
        Tries to guess which channels belong together based on the filename
        alone. It basically works by taking just the basename of the path and
        removing the last letter (after removing all non alphabetic parts) if
        it is E, N or Z and matching now identical filenames.

        This works for the standard SEED notation for waveform data. Other
        formats might work by being lucky but probably not.
        """
        channel_dict = {}
        valid = ['e', 'n', 'z']
        # Use the plain list because it is easier.
        for file in self.plain_list:
            name = os.path.basename(file)
            temp = []
            # Remove all short identifiers in SEED string notation.
            temp_name = name.split('.')
            if len(temp_name) == 1:
                temp_name = name
            else:
                temp_name = [_i for _i in temp_name if len(_i) > 1]
                temp_name = '.'.join(temp_name)
            # Remove all nonalphabetic characters.
            for char in temp_name:
                if char.isalpha():
                    temp.append(char.lower())
            # Do the same once again but keep all numbers!
            digits = []
            for char in temp_name:
                if char.isdigit():
                    digits.append(char)
            digits = ''.join(digits)
            name = ''.join(temp)
            if name[-1] in valid:
                name = name[:-1]
                name = name+str(digits)
                channel_dict.setdefault(name, [])
                channel_dict[name].append(file)
            else:
                channel_dict[file] = [file]
        # Remove all items.
        self.clear()
        # Loop over dictionary and repopulate self.filenames.
        keys = channel_dict.keys()
        keys.sort()
        for key in keys:
            files = channel_dict[key]
            files.sort()
            if len(files) != 3:
                for file in files:
                    size = os.path.getsize(file)
                    string_list = ['%s' % os.path.basename(file), '']
                               #'[1 file - %.2f MB]' % (size/1000000.0)]
                    item = FilenameItem(string_list)
                    sub_item = FilenameItem([file, '%.2f MB' %\
                                             (size/1000000.0)], toplevel=False)
                    item.addChild(sub_item)
                    self.addTopLevelItem(item)
                continue
            name = os.path.basename(files[0])
            temp = name.lower()
            # They are sorted and therefore the east channel should always be
            # first.
            temp = temp.rpartition('e')
            name = name[:len(temp[0])] + name[len(temp[0])+1:]
            size = 0
            for file in files:
                size += os.path.getsize(file)
            string_list = ['%s' % os.path.basename(name),'']
                           #'[3 files - %.2f MB]' % (size/1000000.0)]
            item = FilenameItem(string_list)
            for file in files:
                sub_item = FilenameItem([file, '%.2f MB' %\
                                     (os.path.getsize(file)/1000000.0)],
                                        toplevel=False)
                item.addChild(sub_item)
            self.addTopLevelItem(item)
        # Can't figure out the signal that gets emitted once a new Item is
        # added. So it is done manually.
        self.resizeColumnToContents(0)

    def startEdit(self, item, column):
        """
        Workaround to enable editing only on the first column and all parent
        items.
        """
        parent = item.parent()
        if column == 0 and not parent:
            item.setFlags(QtCore.Qt.ItemIsSelectable |
                QtCore.Qt.ItemIsDropEnabled |
                QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable)
        else:
            if not parent:
                item.setFlags(QtCore.Qt.ItemIsSelectable |
                    QtCore.Qt.ItemIsDropEnabled | QtCore.Qt.ItemIsEnabled)
            else:
                item.setFlags(QtCore.Qt.ItemIsSelectable |
                    QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsEnabled)

    def _connectSignals(self):
        """
        Connect some signals.
        """
        # Disable default double click behaviour.
        self.setExpandsOnDoubleClick(False)
        QtCore.QObject.connect(self,
                   QtCore.SIGNAL('itemDoubleClicked(QTreeWidgetItem *,int)'),
                   self.startEdit)
