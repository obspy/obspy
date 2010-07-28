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
''' HBox class.'''
from operator import add
from container import Container
import theme



class HBox(Container):
    '''
    The HBox class organizes a variable number of widgets into a rectangular
    area which is organized into a single row of child widgets. Thus, all
    children of a horizontal box are allocated a common height.
    '''

    def __init__(self, children=[], homogeneous=True, spacing=1):
        '''
        :Parameters:

        `children` : [glydget.widget.Widget, ...]
            Initial list of children
        `homogeneous` : bool
            If true all children are given equal space allocations.
        `spacing` : int
            The horizontal space between children in pixels.
        '''

        self._spacing = spacing
        self._homogeneous = homogeneous
        Container.__init__(self, children)
        self.style = theme.Container



    def _update_size(self, propagate=False):
        p = self._style.padding
        children = [child for child in self._children if not child._deleted]
        children_nx = [child for child in children if not child._expand[0]]
        children_x = [child for child in children if child._expand[0]]
        if not self._homogeneous:
            hsize = reduce(add,[child._minimum_size[0]
                                for child in children] or [0])
        else:
            hsize = max([child._minimum_size[0]
                         for child in children_x] or [0])*len(children_x)
            hsize += reduce(add,[child._minimum_size[0]
                                for child in children_nx] or [0])
        hsize += (len(children)-1)*self._spacing
        vsize = max([child._minimum_size[1] for child in children] or [0])
        self._minimum_size = [p[2]+p[3]+hsize, p[0]+p[1]+vsize]
        if self.parent and propagate:
            self.parent._update_size(propagate)
        elif propagate:
            self._allocate(self.size_request[0],
                                     self.size_request[1])



    def _allocate(self, width, height):
        Container._allocate(self, width, height)
        style = self._style
        p = style.padding
        content_size = [self._size_allocation[0]-p[2]-p[3],
                        self._size_allocation[1]-p[0]-p[1]]
        x, y = self.x + p[2], self.y - p[0]
        children = [child for child in self._children if not child._deleted]
        children_nx = [child for child in children if not child._expand[0]]
        children_x = [child for child in children if child._expand[0]]

        free = content_size[0] - self._spacing*(len(children)-1)
        free -= reduce(add,[child._minimum_size[0]
                            for child in children_nx] or [0])        
        extra = free - reduce(add,[child.size_request[0]
                                   for child in children_x] or [0])
        child_extra = extra/float(len(children_x))

        if len(children_x):
            common_size = free / float(len(children_x))
        for child in children:
            if not self._homogeneous or not child._expand[0]:
                size = child.size_request[0]
                if child._expand[0]:
                    size +=  child_extra
            else:
                size = common_size
            child._position=[x,y]
            child._allocate(size,content_size[1])
            x += size + self._spacing
