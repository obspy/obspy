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
''' Window '''
import theme
from vbox import VBox
from folder import Folder
from widget import Widget



class Window(Folder):
    ''' Folder '''

    def __init__(self, title='Title', children=None, spacing=3):
        ''' Create folder. '''

        self._vbox = VBox(children=children)
        Folder.__init__(self, title=title, child=self._vbox, active=True, spacing=spacing)
        self.style = theme.Window
        self.title.style = theme.Window.title
        self.child.style = theme.Window.child
        self.title.text = u'✖ '+title
        self._closed_prefix = u'✖ '
        self._opened_prefix = u'✖ '


    def on_mouse_press(self, x, y, button, modifiers):
        if self._deleted:
           return
        self._action = ''
        if not Folder.on_mouse_press(self,x,y,button,modifiers) and self._hit(x,y):
            self._action = 'move'
        if ((x > (self.x+self.width-5)) and
            (x < (self.x+self.width+5)) and
            (y < (self.y-self.height+5)) and
            (y > (self.y-self.height-5))):
            self._action = 'resize'
        return True


    def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
        if self._deleted:
           return
        if self._action == 'move':
            self.move(self.x+dx, self.y+dy)
            return True
        elif self._action == 'resize':
            width = self.width+dx
            height = self.height #-dy
            self.resize(width,0)
            return True
        else:
            return Folder.on_mouse_drag(self,x,y,dx,dy,button,modifiers)

    def on_mouse_motion(self, x, y, dx, dy):
        ''' Default mouse motion handler. '''
        if self._hit(x,y):
            return Folder.on_mouse_motion(self,x,y,dx,dy)
        else:
            if Widget._focused:
                Widget._focused.unfocus()
                Widget._focused = None
