#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# glydget - an OpenGL widget toolkit
# Copyright (c) 2009 - Nicolas P. Rougier
#
# This file is part of glydget.
#
# glydget is free software: you can  redistribute it and/or modify it under the
# terms of  the GNU General  Public License as  published by the  Free Software
# Foundation, either  version 3 of the  License, or (at your  option) any later
# version.
#
# glydget is  distributed in the hope that  it will be useful,  but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy  of the GNU General Public License along with
# glydget. If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
''' Folder '''
import theme
from operator import add
from vbox import VBox
from button import ToggleButton
from widget import Widget


class Folder(VBox):
    ''' Folder '''

    def __init__(self, title='Title', child=None, active=True, spacing=1):
        ''' Create folder. '''

        self._text = title
        self._closed_prefix = u'▹ '
        self._opened_prefix = u'▿ '
        if active:
            prefix = self._opened_prefix
        else:
            prefix = self._closed_prefix
        self._title = ToggleButton(prefix+title, active = active, action=self.toggle)
        VBox.__init__(self, children=[self._title, child], spacing=spacing)
        self.style = theme.Folder
        self.title.style = theme.Folder.title
        self.child.style = theme.Folder.child


    def _build(self, batch=None, group=None):
        ''' Build box and add it to batch and group.

        :Parameters:
            `batch` : `Batch`
                Optional graphics batch to add the object to.
            `group` : `Group`
                Optional graphics group to use.
        '''
        Widget._build(self, batch) #, group)
        self.title._delete()
        self.title._build(batch=self._batch) #, group=self._fg_group)
        if self.child:
            self.child._delete()
        if self.title._active:
            self.child._build(batch=self._batch, group=self._fg_group)
        self._update_state()
        self._update_style()
        self._update_size()



    def toggle(self, *args):
        ''' Toggle active state '''
        if self._title._active:
             self._title.text = self._opened_prefix + self._text
        else:
            self._title.text = self._closed_prefix + self._text
        #self._build(batch=self._batch, group=self._fg_group)
        if self.child:
            self.child._delete()
        if self.title._active:
            self.child._build(batch=self._batch) #, group=self._fg_group)
        self._update_size(True)



    def _get_child(self):
        if len(self._children) == 2:
            return self._children[1]
        else:
            return None

    def _set_child(self, child):
        if len(self._children) == 2:
            self._children[1]._delete()
            self._children[1]._parent = None
            del self._children[1]
        if child:
            self.add(child)

    child = property(_get_child, _set_child, doc='Folder child')



    def _get_title(self):
        return self._title

    title = property(_get_title,
                     doc='Folder title')
                    
