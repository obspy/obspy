#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyglet
from pyglet.gl import *

class WaveformGroup(pyglet.graphics.Group):
    """
    Custom group to handle the vertical scrolling of the different waveforms.

    This will be handled by simply translating the OpenGL view.
    """
    def __init__(self, parent, win, *args, **kwargs):
        """
        Set the parent attribute to still account for the ordering.
        """
        super(WaveformGroup, self).__init__(self, *args, **kwargs)
        self.parent = parent
        self.win = win

    def set_state(self):
        """
        Translate Waveform group.
        """
        glTranslatef(0.0, -self.win.waveform_offset, 0.0)

    def unset_state(self): 
        """
        Undo translation to not affect other elements.
        """
        glTranslatef(0.0, self.win.waveform_offset, 0.0)
        
