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
from label import Label
from button import ToggleButton
import state, theme


class BoolEntry(ToggleButton):
    ''' Text entry. '''

    def __init__(self, value=True, getter=None, setter=None, action=None):
        ''' Create a boolean entry.

        :Parameters:
        `value` : bool
            Initial value
        `getter` : function()
            Function to be used to get actual value
        `setter` : function(value)
            Function to be used to set actual value
        `action` : function(widget)
            Action to be executed when entry has changed
        '''

        self._setter = setter
        self._getter = getter
        self._action_callback = action
        if value:
            text = 'True'
        else:
            text = 'False'
        ToggleButton.__init__(self, text=text, active=value, action=self._on_commit)
        self._expand[0] = True
        if self._getter:
            pyglet.clock.schedule_interval(self._update_value, 0.1)
        self.text = text
        self.style = theme.BoolEntry


    def _on_commit(self, *args):
        if self._setter:
            self._setter(self._active)
        if self._active:
            self.text = 'True'
        else:
            self.text = 'False'
        if self._action_callback:
            self._action_callback(self)

        
    def _update_value(self, dt):
        value = self._getter()
        if value != self._active:
            self._active = value
            if self._active:
                Label.activate(self)
            else:
                Label.deactivate(self)

