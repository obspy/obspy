#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyglet
from status_bar import StatusBar
from background import Background
from scroll_bar import ScrollBar
from status_bar import StatusBar
from waveform_group import WaveformGroup

class PygletWindow(object):
    """
    Class that initializes a windows and handles all loops.
    """
    def __init__(self, env, **kwargs):
        """
        Init Method.
        """
        # Set environment.
        self.env = env
        # Read quarks and set default values.
        self._setDefaultValues(**kwargs)
        # Create the window.
        self.createWindow()
        # Set the caption of the window.
        self.window.set_caption('OpenGL Waveform Database Viewer')
        # Create the ordered groups used to get a layer like drawing of the
        # window.
        self._createdOrderedGroups()
        # Add a background and add it to group 0.
        Background(parent = self, group = 0)
        # Create status bar if desired.
        if self.status_bar:
            # The status bar should always be seen. Add to a very high group.
            self.status_bar = StatusBar(parent = self, group = 999)
        # These cannot be created earlier as they need some already set up GUI
        # Elements.
        # XXX: Need to make more dynamic.
        # Maximum vertical span that is viewable.
        self.max_viewable = 0
        self.current_view_span = self.window.height - self.status_bar.height
        # Add a scroll bar. Should also always be on top.
        self.scroll_bar = ScrollBar(parent = self, group = 999)
        # Create the waveform Group.
        self._createWaveformGroup()


    def getGroup(self, index):
        """
        Gets the group to the corresponding index.

        Return group 0 if it is too small, self. layers if two big and otherwise just
        take the integer value.

        A higher value means that it will be drawn on top of the other layers.
        Very powerful to sort 2D stuff in the OpenGL window.
        """
        index = int(index)
        if index < 0:
            index = 0
        elif index > (self.layers - 1):
            index = (self.layers - 1)
        return self.groups[index]


    def _setDefaultValues(self, **kwargs):
        """
        Central method to collect, read and set all default values.
        """
        # Use to be able to be parent window.
        self.win = self
        self.batch = pyglet.graphics.Batch()
        # List to keep track oj objects.
        self.object_list = []
        # Determines whether or not the window will be resizeable.
        self.resizeable = kwargs.get('resize', True)
        # Vertical synchronisation.
        self.vsync = kwargs.get('vsync', 0)
        # Check whether a status bar is desired.
        self.status_bar = kwargs.get('status_bar', True)
        # Get number of layer. Defaults to five.
        self.layers = kwargs.get('layers', 5)
        # List to store the actual WaveformPlot objects.
        self.waveforms = []
        # Waveform Layer.
        self.waveform_layer = kwargs.get('waveform_layer', 3)
        # Offset of the waveform plots in the y-direction.
        self.waveform_offset = 0


    def _createWaveformGroup(self):
        """
        Creates Waveform Group.

        Needs to be done to get a group that can be automatically passed to
        each waveform group.
        """
        self.waveform_group = WaveformGroup(parent =\
                              self.getGroup(self.waveform_layer), win = self)


    def _createdOrderedGroups(self):
        """
        Using ordered groups enables using sorted vertex list in one single
        batch which then renders as efficiently as possible.
        """
        self.groups = []
        for _i in xrange(self.layers):
            self.groups.append(pyglet.graphics.OrderedGroup(_i))
        

    def createWindow(self):
        """
        Creates the window.
        """
        self.window = \
        pyglet.window.Window(width=800,
                             height=600, resizable=self.resize,
                             vsync=self.vsync, visible = False)
        # Screen properties.
        screen_width = self.window.screen.width
        screen_height = self.window.screen.height
        # Set the size to 90% percent of the screen in either direction.
        self.window.set_size(int(screen_width * 0.9), int(screen_height * 0.9))
        # Set a minium size.
        self.window.set_minimum_size(640, 480)
        # Move to the center of the screen.
        x = (screen_width - int(screen_width * 0.9))//2
        y = (screen_height - int(screen_height * 0.9))//2
        self.window.set_location(x,y) 
        # Finally make the window visible.
        self.window.set_visible()


    def resize(self, width, height):
        """
        Handles all calls necessary on resize.
        """
        # Update vertical span of the window.
        self.current_view_span = height - self.status_bar.height
        # Call the resize method of all objects in the current window.
        for object in self.object_list:
            object.resize(width, height)


    def draw(self, *args, **kwargs):
        """
        Will be called during the main loop.

        The goal is to use only one single batch for maximum efficiency.
        """
        self.window.clear()
        self.batch.draw()

    def mouse_scroll(self, scroll_x, scroll_y):
        """
        Called every time the mouse wheel is scrolled.
        """
        self.waveform_offset += 4 * scroll_y
        if self.waveform_offset > 0:
            self.waveform_offset = 0
        # Avoid going too far down.
        if self.current_view_span - self.waveform_offset > self.max_viewable:
            if self.current_view_span > self.max_viewable:
                self.waveform_offset = 0
            else:
                self.waveform_offset = -((10 + self.max_viewable) - \
                                      self.current_view_span)
        # Update the scroll_bar.
        self.scroll_bar.changePosition()
