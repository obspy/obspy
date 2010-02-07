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
''' Label class. '''
import pyglet
import cached
import theme
from widget import Widget


class Label(Widget):
    '''
    The Label is a widget class that displays a limited amount of read-only
    text. Labels are used by several widgets (e.g. Button) to provide text
    display as well as by applications to display messages to the user.  Most of
    the functionality of a Label is directed at modifying the style and layout
    of the text within the widget allocation. A Label is not focusable nor
    activable by default, which means that it cannot receive events directly.

    Example usage::

        label = Label(text='Hello world !')
        label.show()
    '''

    def __init__(self, text='Label'):
        '''
        :Parameters:

        `text` : str
            Text to be displayed.
        '''

        self._text = text
        self._label = cached.Label(text = self._text)
        Widget.__init__(self)
        self.style = theme.Label


    def _build(self, batch=None, group=None):
        Widget._build(self, batch)
        self._label.delete()
        self._label = cached.Label(text = self._text, multiline=False,
                                  batch=self._batch, group=self._fg_group,
                                  anchor_x='left', anchor_y='top')
        self._deleted = False
        self._update_style()
        self._update_state()
        self._update_size()


    def _delete(self):
        Widget._delete(self)
        self._label.delete()


    def _move(self, x, y):
        dx, dy = x-self.x, y-self.y
        Widget._move(self,x,y)
        widget = self._label
        widget.x = widget.x+dx
        widget.y = widget.y+dy


    def _update_size(self, propagate=False):
        p = self._style.padding
        self._minimum_size = [p[2]+p[3]+self._label.content_width,
                              p[0]+p[1]+self._label.content_height]
        if self.parent and propagate:
            self.parent._update_size(propagate)
        elif propagate:
            self._allocate(self.size_request[0],
                                     self.size_request[1])



    def _allocate(self, width, height):
        Widget._allocate(self, width, height)
        if self._deleted:
            return
        label = self._label
        style = self._style
        p = style.padding
        halign = style.halign
        valign = style.valign
        content_width = self.width-p[2]-p[3]
        content_height = self.height-p[0]-p[1]
        label.x = int(self.x+p[2]+(content_width-label.content_width)*halign)
        label.y = int(self.y-p[0]-(content_height-label.content_height)*valign)



    def _update_state(self):
        if self._deleted:
            return
        Widget._update_state(self)
        self._label.color = self._style.colors[self._state]


    def _update_style(self):
        if self._deleted:
            return
        style = self._style
        self._label.font_name = style.font_name
        self._label.font_size = style.font_size
        self._label.bold = style.bold
        self._label.italic = style.italic
        Widget._update_style(self)



    def _get_text(self):
        return self._label.text

    def _set_text(self, text):
        self._text = text
        self._label.text = text

    text = property(_get_text, _set_text,
                    doc='''
    Displayed text

    :type: str, read-write.
    ''')
