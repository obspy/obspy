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
''' Container class. '''
from operator import add
from widget import Widget
import state, theme



class Container(Widget):
    '''
    The Container class is an abstract class providing common attributes and
    methods for a number of widget subclasses that manage the layout of other
    widgets within the area of a window.
    '''

    def __init__(self, children=[]):
        '''
        :Parameters:

        `children` : [:class:`glydget.widget.Widget`, ...]
            Initial list of children to be added to the container.
        '''

        self._children = children
        for child in self._children:
            child._parent = self
            child._update_size()
        Widget.__init__(self)
        self.style = theme.default



    def insert(self, child, index=0):
        ''' Insert a new child into the container.

        Inserting a child into a container usually results in the resizing and
        redrawing of the container contents.

        :Parameters:

        `index` : int
            index of place where to insert child

        `child` : glydget.Widget
            child to be added
        '''
        if child._parent:
            child._parent.remove(child)
        self._children.insert(index, child)
        child._update_size()
        child._parent = self
        child._build(self._batch) #, self._fg_group)
        self._update_size(True)
        root = self.root
        root._allocate(root.width, root.height)
        


    def prepend(self, child):
        ''' Prepend a child to the container.

        Prepending a child to a container usually results in the resizing and
        redrawing of the container contents.

        :Parameters:

        `child` : glydget.Widget
            child to be added
        '''

        self.insert(child, 0)



    def append(self, child):
        ''' Append a child to the container.

        Appending a child to a container usually results in the resizing and
        redrawing of the container contents.

        :Parameters:

        `child` : glydget.Widget
            child to be added
        '''

        self.insert(child, len(self._children))



    def remove(self, child):
        ''' Remove a child from container.

        Removing a child to a container usually results in the resizing and
        redrawing of the container contents.

        :Parameters:

        `child` : glydget.Widget
            child to be removed (must be currently in container).
        '''

        if not child in self._children:
            return
        index = self._children.index(child)
        self._children[index]._parent = None
        self._children[index]._delete()
        del self._children[index]
        self._update_size(True)
               

       
    def _get_children(self):
        return self._children

    children = property(_get_children,
                       doc = '''
    Container's children.

    :type: [`glydget.widget.Widget`,...], read-only.
    ''')


    def __getitem__(self, i):
        ''' Get i-th child.

        :Parameters:

        `i` : int
            `index` of child to get.
        '''

        return self.children(i)



    def __setitem__(self, i, child):
        ''' Set i-th child.

        :Parameters:

        `i` : int
            index of child to be set.
        `child` : glydget.Widget
            child to replace i-th child
        '''
        
        self.remove(self.children(i))
        self.insert(child,i)



    def __delitem__(self, i):
        ''' Delete i-th child.

        :Parameters:

        `i` : int
            index of child to be deleted.
        '''

        self.remove(self.children(i))



    def _build(self, batch=None, group=None):

        Widget._build(self, batch) #, group)
        for child in self._children:
            child._delete()
            child._build(batch=self._batch) #, group=self._fg_group)
        self._deleted = False
        self._update_style()
        self._update_state()
        self._update_size()



    def _delete(self):
        ''' Delete container. '''

        Widget._delete(self)
        for child in self._children:
            child._delete()



    def _move(self, x, y):

        dx, dy = x-self.x, y-self.y
        Widget._move(self,x,y)
        for child in self._children:
            child._move(child.x+dx, child.y+dy)



    def _update_state(self):

        if self._deleted:
            return
        Widget._update_state(self)
        for child in self._children:
            child._update_state()



    def on_mouse_press(self, x, y, button, modifiers):
        ''' Default mouse press hanlder. '''

        for child in self._children:
            if not child._deleted and child._hit(x,y):
                return child.on_mouse_press(x, y, button, modifiers)



    def on_mouse_release(self, x, y, button, modifiers):
        ''' Default mouse release hanlder. '''

        for child in self._children:
            if not child._deleted and child._hit(x,y):
                return child.on_mouse_release(x, y, button, modifiers)



    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        ''' Default mouse scroll handler. '''

        for child in self._children:
            if not child._deleted and child._hit(x,y):
                return child.on_mouse_scroll(x, y, scroll_x, scroll_y)



    def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
        ''' Default mouse drag hanlder. '''

        for child in self._children:
            if not child._deleted and child._hit(x,y):
                return child.on_mouse_drag(x, y, dx, dy, button, modifiers)



    def on_mouse_motion(self, x, y, dx, dy):
        ''' Default mouse motion hanlder. '''

        for child in self._children:
            if not child._deleted and child._hit(x,y):
                return child.on_mouse_motion(x, y, dx, dy)



    def on_key_press(self, symbol, modifiers):
        ''' Default key press handler. '''

        widget = Widget._focused or Widget._activated
        if widget:
            widget.on_key_press(symbol, modifiers)



    def on_key_release(self, symbol, modifiers):
        ''' Default key release handler. '''

        widget = Widget._focused or Widget._activated
        if widget:
            widget.on_key_release(symbol, modifiers)

    def on_text(self, text):
        ''' Default text hanlder. '''

        widget = Widget._activated
        if widget:
            widget.on_text(text)



    def on_text_motion(self, motion):
        ''' Default text motion hanlder. '''
        widget = Widget._activated
        if widget:
            widget.on_text_motion(motion)



    def on_text_motion_select(self, motion):
        ''' Default text motion select hanlder. '''

        widget = Widget._activated
        if widget:
            widget.on_text_motion_select(motion)


