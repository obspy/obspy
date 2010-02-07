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
''' Scalar entry. '''
import sys
import pyglet
import pyglet.window.key as key
from entry import Entry
import state, theme


class TextEntry(Entry):
    ''' Text entry. '''

    def __init__(self, text='text', getter=None, setter=None, on_change=None):
        ''' Create a text entry.

        :Parameters:
        `text` : str
            Initial text
        `getter` : function()
            Function to be used to get actual value
        `setter` : function(value)
            Function to be used to set actual value
        `on_change` : function(entry)
            Function to be called when entry has changed
        '''

        self._text = text
        self._setter = setter
        self._getter = getter
        self._on_change = on_change
        Entry.__init__(self, text=text, on_change=self._on_commit)
        self._expand[0] = True
        if self._getter:
            pyglet.clock.schedule_interval(self._update_text, 0.1)
        self.text = text



    def _on_commit(self, *args):
        if self._setter:
            self._setter(self.text)
        if self._on_change:
            self._on_change(self)

        
    def _update_text(self, dt):
        if self._getter and not self._active:
            text = self._getter()
            if text != self.text:
                self.text = text

