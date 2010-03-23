#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyglet
from pyglet.gl import *

class WaveformGroup(pyglet.graphics.Group):
    """
    Custom group to handle the vertical scrolling of the different waveforms.

    This will be handled by simply translating the OpenGL view.
    """
    def __init__(self, win, x_offset, y_offset, id, plot = False,
                 plot_offset = 0.0, *args, **kwargs):
        """
        Set the parent attribute to still account for the ordering.
        """
        super(WaveformGroup, self).__init__(self, *args, **kwargs)
        self.win = win
        # The parent is a ordered group that gets handled by the window which
        # also decided which layer the waveforms will be drawn on.
        if id == 1:
            self.parent = self.win.getGroup(self.win.waveform_layer1)
        elif id == 2:
            self.parent = self.win.getGroup(self.win.waveform_layer2)
        elif id == 3:
            self.parent = self.win.getGroup(self.win.waveform_layer3)
        # Constant offsets.
        self.x_offset = x_offset
        # The given offset is always the offset from the top border of the
        # window.
        self.y_offset = y_offset
        self.current_offset = self.win.window.height - self.y_offset
        # If self.plot is True an additional gScalef will be called.
        self.plot = plot
        self.plot_offset = plot_offset

    def set_state(self):
        """
        Translate Waveform group.
        """
        if self.plot:
            self.current_offset = self.win.window.height - self.y_offset
            glTranslatef(self.x_offset + self.plot_offset,
                         self.current_offset -self.win.waveform_offset -\
                         self.win.geometry.time_scale, 0.0)
            glScalef(self.plot, 1.0, 1.0)
        else:
            self.current_offset = self.win.window.height - self.y_offset
            glTranslatef(self.x_offset, self.current_offset -\
                         self.win.waveform_offset - self.win.geometry.time_scale, 0.0)

    def unset_state(self): 
        """
        Undo translation to not affect other elements.
        """
        if self.plot:
            glScalef(1.0/self.plot, 1.0, 1.0)
            self.current_offset = self.win.window.height - self.y_offset
            glTranslatef(-self.x_offset - self.plot_offset,
                         -self.current_offset +self.win.waveform_offset +\
                         self.win.geometry.time_scale, 0.0)
        else:
            self.current_offset = self.win.window.height - self.y_offset
            glTranslatef(-self.x_offset, -self.current_offset +\
                         self.win.waveform_offset + \
                         self.win.geometry.time_scale, 0.0)
