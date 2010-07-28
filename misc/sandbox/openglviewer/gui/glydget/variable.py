#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# glydget - an OpenGL widget toolkit
# Copyright (c) 2009 - Nicolas P. Rougier
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
''' Variable '''
import sys
import string
from hbox import HBox
from label import Label
from scalar_entry import ScalarEntry
from bool_entry import BoolEntry
from text_entry import TextEntry


class Variable(HBox):

    def __init__(self, name, label = None,
                 vmin=None, vmax=None, vstep=None, vformat=None,
                 getter=None, setter=None, action=None,
                 namespace=None, read_only=False):
        '''
        Create a new variable.

        :Parameters:
        
        `name` : str
            Name of the variable
        `label` : str
            Label to be displayed next to variable value
        `vmin` : float
            Minimum value of variable (only for scalar variables)
        `vmax` : float
            Maximum value of variable (only for scalar variables)
        `vstep` : float
            Value step increment (only for scalar variables)
        `vformat` : str
            Variable representatin format
        `getter` : function()
            Function to be used to get actual value
        `setter` : function(value)
            Function to be used to set actual value
        `action` : function(widget)
            Action to be executed when entry has changed
        '''
        self._name = name
        self._namespace = namespace
        self._read_only = read_only
        self._getter = getter
        self._setter = setter
        self._action = action
        self._type = None
        if '.' in name:
            self._object = eval(string.join(name.split('.')[:-1],'.'),namespace)
            self._attribute = name.split('.')[-1]
            self._type = type(getattr(self._object, self._attribute))
        else:
            self._object = None
            self._attribute = None
            self._type = type(namespace[name])

        # Create dedicated entry
        if self._type == bool:
            self._entry = BoolEntry(value=self.get(), action=action,
                                    getter=self.get, setter=self.set)
        elif self._type == int:
            if vmin == None:
                vmin = -sys.maxint-1
            if vmax == None:
                vmax = sys.maxint
            vstep = vstep or 1
            vformat = vformat or '%d'
            self._entry = ScalarEntry(value=self.get(), action=action,
                                      getter=self.get, setter=self.set,
                                      vmin=vmin, vmax=vmax, vstep=vstep, vformat=vformat)
        elif self._type == float:
            if vmin == None:
                vmin = -sys.maxint-1
            if vmax == None:
                vmax = sys.maxint
            vstep = vstep or 0.1
            vformat = vformat or '%.2f'
            self._entry = ScalarEntry(self.get(), action=action,
                                      getter=self.get, setter=self.set,
                                      vmin=vmin, vmax=vmax, vstep=vstep,vformat=vformat) 
        elif self._type == str:
            self._entry = TextEntry(self.get(), self.get, self.set, action=action)
        else:
            self._entry = Label(str(self.get()))

        if self._read_only:
            self._entry.focusable = False
            self._entry.activable = False

        if label:
            self._label = Label(label)
        else:
            self._label = Label(name)        
        HBox.__init__(self, [self._label, self._entry], homogeneous=True)



    def _get_name(self):
        return self._name
    name = property(_get_name,
                    doc = '''
    Variable name

    :type: str, read-only.
    ''')


    def set(self, value):
        ''' Set value '''
        
        if self._read_only:
            return
        try:
            if self._setter:
                self._setter(value)
            else:
                if self._object:
                    setattr(self._object, self._attribute, value)
                else:
                    self._namespace[self._name] = value
        except:
            self._read_only = True
            self._entry._focusable = False
            self._entry._activable = False



    def get(self):
        ''' Get value '''

        if self._getter:
            return self._getter()
        else:
            if self._object:
                return getattr(self._object, self._attribute)
            else:
                return self._namespace[self._name]
