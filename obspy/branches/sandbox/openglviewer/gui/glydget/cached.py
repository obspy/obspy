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
''' pyglet cached label. '''
import pyglet


class Label(pyglet.text.Label):
    ''' Pyglet label replacement using cached groups. '''

    _cached_groups = {}
    def _init_groups(self, group):
        if not group:
            return
        if group not in self.__class__._cached_groups.keys():
            top = pyglet.text.layout.TextLayoutGroup(group)
            bg = pyglet.graphics.OrderedGroup(0,top)
            fg = pyglet.text.layout.TextLayoutForegroundGroup(1,top)
            fg2 = pyglet.text.layout.TextLayoutForegroundDecorationGroup (2,top)
            self.__class__._cached_groups[group] = [top,bg,fg,fg2,0]
        groups = self.__class__._cached_groups[group]
        self.top_group= groups[0]
        self.background_group = groups[1] 
        self.foreground_group = groups[2]
        self.foreground_decoration_group = groups[3]
        groups[4] += 1

    def delete(self):
        pyglet.text.Label.delete(self)
        if self.top_group and self.top_group.parent:
            group = self.top_group.parent
            if group is not None:
                groups = self.__class__._cached_groups[group]
                groups[4] -= 1
                if not groups[4]:
                    del self.__class__._cached_groups[group]
        self.top_group = None
        self.background_self = None
        self.foreground_group = None
        self.foreground_decoration_group = None
