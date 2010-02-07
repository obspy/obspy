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
from vbox import VBox
from hbox import HBox
from button import Button
import state, theme


class ScalarEntry(HBox):
    ''' Scalar entry. '''

    def __init__(self, value=0, vmin=-sys.maxint-1, vmax=sys.maxint, vstep=.1,
                 vformat='%.2f', key_increase=key.UP, key_decrease=key.DOWN,
                 getter=None, setter=None, action=None):
        ''' Create a numeric entry.

        :Parameters:
        `value` : scalar
            Initial value
        `vmin` : scalar
            Minimum value
        `vmax` : scalar
            Maximum value
        `vstep` : scalar
            Value step increment
        `vformat` : str
            Representation string format
        `key_increase` : char
            Key binding for increasing value
        `key_decrease` : char
            Key binding for decreasing value
        `getter` : function()
            Function to be used to get actual value
        `setter` : function(value)
            Function to be used to set actual value
        `action` : function(widget)
            Action to be executed when entry has changed
        '''
        self._value = value
        self._vmin = vmin
        self._vmax = vmax
        self._vstep = vstep
        self._vformat = vformat
        self._key_increase = key_increase
        self._key_decrease = key_decrease
        self._setter = setter
        self._getter = getter
        self._action = action
        self._button_up = Button(u'⬆', action=self.increase_value)
        self._button_up._expand[0] = False
        self._button_up.style = theme.ArrowButton
        self._button_down = Button(u'⬇', action=self.decrease_value)
        self._button_down._expand[0] = False
        self._button_down.style = theme.ArrowButton
        self._buttons = VBox([self._button_up, self._button_down], homogeneous=True)
        self._buttons._expand[0] = False
        self._entry = Entry(action=self._on_commit)
        self._entry._expand[0] = True
        HBox.__init__(self, [self._entry, self._buttons], homogeneous=True)
        if self._getter:
            pyglet.clock.schedule_interval(self._update_value, 0.1)
        self.value = value



    def _get_focusable(self):
        return self._entry.focusable

    def _set_focusable(self, focusable):
        self._button_up.focusable = focusable
        self._button_down.focusable = focusable
        self._entry.focusable = focusable

    focusable = property(_get_focusable, _set_focusable,
                         doc = ''' Indicate whether widget can be focused. ''')


    def _get_activable(self):
        return self._entry.activable

    def _set_activable(self, activable):
        self._button_up.activable = activable
        self._button_down.activable = activable
        self._entry.activable = activable

    activable = property(_get_activable, _set_activable,
                         doc = ''' Indicate whether widget can be activated. ''')

        
    def increase_value(self, button=None):
        ''' Increase value '''
        self.value += self._vstep
        self._on_commit()


    def decrease_value(self, button=None):
        ''' Decrease value '''
        self.value -= self._vstep
        self._on_commit()


    def _update_value(self, dt):
        if self._getter and not self._entry._active:
            value = self._getter()
            if value != self.value:
                self._value = max(min(value, self._vmax),self._vmin)
                self._entry._text = self._vformat % self._value
                self._entry.text = self._entry._text


    def _on_commit(self, *args):
        value = self._value
        try:
            self.value = float(self._entry.text)
        except:
            self.value = value
        self._entry.text = self._vformat % self._value
        if self._action:
            self._action(self)


    def _get_value(self):
        return self._value

    def _set_value(self, value):
        self._value = max(min(value, self._vmax),self._vmin)
        self._entry._text = self._vformat % self._value
        self._entry.text = self._entry._text
        if self._setter:
            self._setter(self._value)

    value = property(_get_value, _set_value,
                     doc='Represented value')


    def _get_vmin(self):
        return self._vmin

    def _set_vmin(self, vmin):
        self._vmin = vmin
        self.value = self.value

    vmin = property(_get_vmin, _set_vmin,
                     doc='Minimum value')


    def _get_vmax(self):
        return self._vmax
    def _set_vmax(self, vmax):
        self._vmax = vmax
        self.value = self.value
    vmax = property(_get_vmax, _set_vmax,
                     doc='Maximum value')


    def _get_vstep(self):
        return self._vstep

    def _set_vstep(self, vstep):
        self._vstep = vstep

    vstep = property(_get_vstep, _set_vstep,
                     doc='Value step increment')


    def _get_vformat(self):
        return self._vformat

    def _set_vformat(self, vformat):
        self._vformat = vformat

    vformat = property(_get_vformat, _set_vformat,
                       doc='Value representation format string')


    def _get_key_increase(self):
        return self._key_increase

    def _set_key_increase(self, key_increase):
        self._key_increase = key_increase

    key_increase = property(_get_key_increase, _set_key_increase,
                            doc='Key binding for increasing value')


    def _get_key_decrease(self):
        return self._key_decrease

    def _set_key_decrease(self, key_decrease):
        self._key_decrease = key_decrease

    key_decrease = property(_get_key_decrease, _set_key_decrease,
                            doc='Key binding for decreasing value')


    def _get_getter(self):
        return self._getter

    def _set_getter(self, getter):
        self._getter = getter

    getter = property(_get_getter, _set_getter,
                      doc='''Function to be used to get value

    :type: function()
    ''')


    def _get_setter(self):
        return self._setter

    def _set_setter(self, setter):
        self._setter = setter

    setter = property(_get_setter, _set_setter,
                      doc='''Function to be used to set value

    :type: function(value)
    ''')



    def _get_action(self):
        return self._action

    def _set_action(self, action):
        self._action = action

    action = property(_get_action, _set_action,
             doc = '''Action to executed when button is pressed

    :type: function(button)
    ''')


    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        ''' Mouse scroll handler

        Scroll up   : increase value.
        Scroll down : decrease value.
        '''
        if self._deleted or not self._hit(x,y):
            return
        self.value += scroll_y*self._vstep
        self._on_commit()
        return True


    def on_key_press(self, symbol, modifiers):
        ''' Key press handler (needs focus)

        Return or Enter: set value
        '+' or self.key_increase : increase value
        '-' or self.key_decrease : decrease value
        '''
        if self._deleted or self._state not in [state.focused, state.activated]:
            return
        if symbol == self._key_decrease:
            self.value -= self._vstep
            self._on_entry_change()
            return True
        elif symbol == self._key_increase:
            self.value += self._vstep
            self._on_commit()
            return True
