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
''' Button classes. '''
from label import Label
import state, theme


class Button(Label):
    '''
    The Button widget is usually displayed as a push button with a text label
    and is used to execute an action when it is activated.

    Example usage::

        def click(button):
            print 'You clicked me !'
        button = Button(text='Click me', action=click)
        button.show()
        window.push_handlers(button)
    '''

    def __init__(self, text='Label', action=None):
        '''
        :Parameters:

        `text` : str
            Text to be displayed within button
        `action` : function(button)
            Action to be executed when button is toggled
        '''

        Label.__init__(self, text)
        self._focusable = True
        self._activable = True
        self._action = action
        self.style = theme.Button



    def activate(self):
        ''' Activate button. '''

        Label.activate(self)
        if self._action:
            self._action(self)
        Label.deactivate(self)



    def _get_action(self):
        return self._action

    def _set_action(self, action):
        self._action = action

    action = property(_get_action, _set_action,
             doc = '''Action to executed when button is activated.

    :type: function(button)
    ''')




class ToggleButton(Button):
    '''
    A toggle button is displayed as a push button with a text label. When button
    is toggled, an optional action can be executed.
    '''

    def __init__(self, text='Label', active=False, action=None):
        '''
        :Parameters:

        `text` : str
            Text to be displayed within button
        `action` : function(button)
            Action to be executed when button is toggled
        `toggled` : bool
            Initial state of the toggle button
        '''

        Button.__init__(self, text, action)
        self._active = active
        if self._active:
            self._state = state.activated



    def activate(self):
        ''' Activate toggle button. '''

        self._active = not self._active
        Button.activate(self)



    def _update_state(self):
        ''' Update widget state '''

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
                


    def _get_active(self):
        return self._active

    def _set_active(self, active):
        if active != self._active:
            self.activate()

    active = property(_get_active, _set_active,
             doc = '''Button active state.

    :type: bool
    ''')
