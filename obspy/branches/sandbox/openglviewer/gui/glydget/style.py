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
''' Style class. '''
import copy


class Style(object):
    '''
    A Style describes the appearance of a :class:`glydget.widget.Widget` for
    each of its possible state.
    '''

    def __init__(self):
        '''
        '''
        self.colors = [[255,]*4]*5
        self.foreground_colors = [[0,]*4*8]*5
        self.background_colors = [[0,]*4*4]*5
        self.font_name = 'Gill Sans'
        self.font_size = 10
        self.bold = False
        self.italic = False
        self.padding = [2,2,2,2] # Up, Down, Left, Right
        self.halign = 0.0
        self.valign = 0.5


    def _get_padding(self):
        return self._padding

    def _set_padding(self, padding):
        self._padding = padding

    padding = property(_get_padding, _set_padding,
                      doc = '''Padding.

    Padding defines the space between widget up, down, left, right borders and
    widget contents.

    :type: [int,int,int,int]
    ''')


    def _get_font_name(self):
        return self._font_name

    def _set_font_name(self, font_name):
        self._font_name = font_name

    font_name = property(_get_font_name, _set_font_name,
                      doc = '''Font family name.

    :type: str or list
    ''')


    def _get_font_size(self):
        return self._font_size

    def _set_font_size(self, font_size):
        self._font_size = font_size

    font_size = property(_get_font_size, _set_font_size,
                      doc = '''Font size, in points.

    :type: float
    ''')


    def _get_bold(self):
        return self._bold

    def _set_bold(self, bold):
        self._bold = bold

    bold = property(_get_bold, _set_bold,
                    doc = '''Font boldness.

    :type: bool
    ''')


    def _get_italic(self):
        return self._italic

    def _set_italic(self, italic):
        self._italic = italic

    italic = property(_get_italic, _set_italic,
                      doc = '''Font slant.

    :type: bool
    ''')


    def _get_colors(self):
        return self._colors

    def _set_colors(self, colors):
        self._colors = colors

    colors = property(_get_colors, _set_colors,
                      doc = '''Text colors

    A text color corresponds to a list of RGBA components in between 0 and
    255. Since a widget can have several states, such a list must be specified
    for each of the different states (currently, 5).

    :type: [[int,]*4]*5  (see explanations above).
    ''')



    def _get_foreground_colors(self):
        return self._foreground_colors

    def _set_foreground_colors(self, colors):
        self._foreground_colors = colors

    foreground_colors = property(_get_foreground_colors, _set_foreground_colors,
                                 doc = '''Foreground colors

    Any widget foreground is made of an outlined rectangle where line colors can
    be specified independently. A foreground color is then actually made of 8
    RGBA colors specifying the top line (left and right colors), bottom line
    (left and right colors), right line (top and bottom colors) and left line
    (top and bottom colors).  These components must be given as a flat list of
    8x4 integers representing 8 RGBA components between 0 and 255. Furthermore
    and since a widget can have several states, such a list must be specified
    for each of the different states (currently, 5).

    :type: [[int,]*4*8,]*5 (see explanations above).
    ''')



    def _get_background_colors(self):
        return self._background_colors

    def _set_background_colors(self, colors):
        self._background_colors = colors

    background_colors = property(_get_background_colors, _set_background_colors,
                                 doc = '''Background colors

    Any widget background is made of a filled rectangle where corner colors can
    be specified independently. A background color is then actually made of 4
    RGBA colors specifying the top-left, top-right, bottom-left and bottom-right
    corners. These components must be given as a flat list of 4x4 integers
    representing 4 RGBA components between 0 and 255. Furthermore and since a
    widget can have several states, such a list must be specified for each of
    the different states (currently, 5).

    :type: [[int,]*4*4,]*5 (see explanations above).
    ''')
