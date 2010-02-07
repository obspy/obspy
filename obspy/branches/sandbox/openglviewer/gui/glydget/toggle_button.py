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
from widget import Widget
from label import Label
from button import Button
import state, theme


class ToggleButton(Button):

    def __init__(self, text='Label', active=False, on_click=None):
        ''' Create toggle button. '''

        self._active = active
        Button.__init__(self, text, on_click)
        if self._active:
            self._state = state.activated



    def activate(self):
        ''' Activate toggle button. '''

        Label.activate(self)
        if not self._active:
            self._active = True
        else:
            self._active = False
        if self._on_click:
            self._on_click(self)
        Label.deactivate(self)


    def _update_state(self):

        if self._deleted:
            return           
        s = self._state
        if s == state.default and self._active:
            s = state.activated
        self._label.color = self._style.colors[self._state]
        self._foreground._colors = self._style.foreground_colors[s]
        self._foreground._update_colors()
        self._background._colors = self._style.background_colors[s]
        self._background._update_colors()
                
