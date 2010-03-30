#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyglet
from pyglet.window import mouse
import glydget
from menu import Menu
from obspy.core import UTCDateTime
from status_bar import StatusBar
from background import Background
from scroll_bar import ScrollBar
from timers import Timers
from seishub import Seishub
from time_scale import TimeScale
from utils import Utils

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
        # Create geometry instance.
        self.geometry = Geometry()
        geo = self.geometry
        # Read quarks and set default values.
        self._setDefaultValues(**kwargs)
        # Create the window.
        self.createWindow()
        # Set the OpenGL state.
        self.setOpenGLState()
        # Set the caption of the window.
        self.window.set_caption('OpenGL Waveform Database Viewer')
        # Create the ordered groups used to get a layer like drawing of the
        # window.
        self._createdOrderedGroups()
        # Add a background and add it to group 0.
        Background(parent = self, group = 0)
        # Connect to SeisHub Server.
        self.seishub = Seishub(parent = self, group = 1)
        # Create status bar if desired.
        if self.status_bar:
            # The status bar should always be seen. Add to a very high group.
            self.status_bar = StatusBar(parent = self, group = 999,
                                height = self.geometry.status_bar_height,
                                error = self.default_error)
        # Create Utils class.
        self.utils = Utils(parent = self, group = 1)
        # Add the Time Scale.
        self.time_scale = TimeScale(parent = self, group = 999)
        # These cannot be created earlier as they need some already set up GUI
        # Elements.
        # XXX: Need to make more dynamic.
        # Maximum vertical span that is viewable.
        self.max_viewable = 0
        self.current_view_span = self.window.height - self.status_bar.height
        # Add a scroll bar. Should also always be on top.
        self.scroll_bar = ScrollBar(parent = self, group = 999)
        # Add the menu.
        self.menu = Menu(parent = self, group = 999, width = self.geometry.menu_width)
        geo.menu_width = self.menu.menu.width
        # Start of menu.
        self.menu_start = self.window.width - (geo.menu_width +\
                    geo.horizontal_margin + geo.scroll_bar_width)
        # Preload some cursors for faster access.
        self.default_cursor = \
                        self.window.get_system_mouse_cursor(self.window.CURSOR_DEFAULT)
        self.hand_cursor = self.window.get_system_mouse_cursor(self.window.CURSOR_HAND)
        # Init zoom box handler.
        self.zoomBox()
        # Start timers.
        self.Timers = Timers(parent = self, group = 1)

    def setOpenGLState(self):
        """
        Sets some global OpenGL states.
        """
        # Enable transparency.
        pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA,
                              pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)


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
            return self.top_group1
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
        # Get number of layer. Defaults to six.
        self.layers = kwargs.get('layers', 6)
        # Number of bars for each waveform plot.
        self.detail = 1000
        # Logarithmic scale.
        self.log_scale = 10
        # List to store the actual WaveformPlot objects.
        self.waveforms = []
        # Start- and Endtime of the plots. Needs to be stored into the window
        # object because it has to be the same for all traces.
        self.starttime = kwargs.get('starttime', UTCDateTime(2010,1,1))
        self.endtime = kwargs.get('endtime', UTCDateTime(2010,2,19) - 1.0)
        # Waveform Layer. Waveforms will need some more layers. I will just add
        # three.
        # XXX: Maybe switch to some more elegant solution.
        self.waveform_layer1 = kwargs.get('waveform_layer', 3)
        self.waveform_layer2 = self.waveform_layer1 + 1
        self.waveform_layer3 = self.waveform_layer2 + 1
        # Offset of the waveform plots in the y-direction.
        # XXX: Currently used?
        self.waveform_offset = 0
        # Zoom box.
        self.zoom_box = None
        # Default error.
        self.default_error = ''

    def _createdOrderedGroups(self):
        """
        Using ordered groups enables using sorted vertex list in one single
        batch which then renders as efficiently as possible.
        """
        self.groups = []
        for _i in xrange(self.layers):
            self.groups.append(pyglet.graphics.OrderedGroup(_i))
        # Create one top level group. Useful for dialog boxes and other stuff
        # that goes over everything else.
        self.top_group1 = pyglet.graphics.OrderedGroup(99)
        self.top_group2 = pyglet.graphics.OrderedGroup(9999)
        

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
        # Set the size to 85% percent of the screen in either direction.
        self.window.set_size(int(screen_width * 0.85), int(screen_height * 0.85))
        # Set a minium size.
        self.window.set_minimum_size(640, 480)
        # Move to the center of the screen.
        x = (screen_width - int(screen_width * 0.85))//2
        y = (screen_height - int(screen_height * 0.85))//2
        self.window.set_location(x,y) 
        # Finally make the window visible.
        self.window.set_visible()


    def resize(self, width, height):
        """
        Handles all calls necessary on resize.
        """
        geo = self.geometry
        # Start of menu.
        self.menu_start = self.window.width - (geo.menu_width +\
                    geo.horizontal_margin + geo.scroll_bar_width)
        # Update vertical span of the window.
        self.current_view_span = height - self.status_bar.height
        # Call the resize method of all objects in the current window.
        for object in self.object_list:
            object.resize(width, height)
        # Just one call to the adaptive plot height is needed. Therefore the
        # calls need to be here.
        if self.waveforms:
            self.utils.adaptPlotHeight()

    def draw(self, *args, **kwargs):
        """
        Will be called during the main loop.

        The goal is to use only one single batch for maximum efficiency.
        """
        self.window.clear()
        self.batch.draw()
        
    def zoomBox(self):
        """
        Creates zoom box.
        """
        def on_mouse_motion(x, y, dx, dy):
            """
            Called every time the mouse moves.
            """
            if in_box(x, y):
                # Change the cursor if inside the box.
                self.window.set_mouse_cursor(self.hand_cursor)
            else:
                self.window.set_mouse_cursor(self.default_cursor)

        def on_mouse_press(x, y, button, modifiers):
            """
            Called on a mouse button press.
            """
            geo = self.geometry
            if in_box(x, y):
                if button & mouse.LEFT:
                    # Some variables.
                    start = geo.horizontal_margin + geo.graph_start_x
                    end =  self.window.width - 2 * geo.horizontal_margin - \
                            geo.scroll_bar_width - geo.menu_width - 3
                    span = end - start
                    # Time calculations.
                    time_span = self.win.endtime - self.win.starttime
                    starttime = self.win.starttime + float(self.zoom_box_min_x -\
                            start)/span * time_span
                    endtime = self.win.starttime + float(self.zoom_box_max_x -\
                            start)/span * time_span
                    self.utils.changeTimes(starttime, endtime)
                    delete_and_pop_handlers()
                # If the right button is pressed, delete the box.
                elif button & mouse.RIGHT:
                    delete_and_pop_handlers()

        def in_box(x, y):
            """
            Checks whether the mouse is in the box.
            """
            if self.zoom_box and x <= self.zoom_box_max_x and \
            x >= self.zoom_box_min_x and y >= self.zoom_box_min_y and \
            y <= self.zoom_box_max_y:
                return True
            else:
                return False

        def delete_and_pop_handlers():
            """
            Deletes the box, the frame and pops the top handlers from the
            stack.
            """
            self.zoom_box.delete()
            self.zoom_frame.delete()
            self.zoom_box = None
            # Popping handlers.
            # XXX: Are these always the right handlers??
            self.win.window.pop_handlers()
            # Return to the default cursor.
            self.window.set_mouse_cursor(self.default_cursor)

        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            """
            Actual handler needed for pushing.
            """
            geo = self.geometry
            height = self.window.height - 2 * geo.vertical_margin - 3 -\
                geo.status_bar_height
            start = geo.horizontal_margin + geo.graph_start_x
            end =  self.window.width - 2 * geo.horizontal_margin - \
                    geo.scroll_bar_width - geo.menu_width - 3
            # If the box already exists, update it.
            if self.zoom_box:
                if x < start:
                    x = start
                elif x > end:
                    x = end
                self.zoom_box.begin_update()
                self.zoom_box.resize(x - self.zoom_start - 2, height - 2)
                self.zoom_box.end_update()
                self.zoom_frame.begin_update()
                self.zoom_frame.resize(x - self.zoom_start, height)
                self.zoom_frame.end_update()
                self.zoom_box_min_x = self.zoom_start
                self.zoom_box_max_x = self.zoom_start + x - self.zoom_start
                self.zoom_box_max_y = self.window.height - geo.vertical_margin\
                                      - 3
                self.zoom_box_min_y = self.zoom_box_max_y - height
                if self.zoom_box_min_x > self.zoom_box_max_x:
                    self.zoom_box_min_x, self.zoom_box_max_x = \
                    self.zoom_box_max_x, self.zoom_box_min_x
            # Otherwise create a new box.
            else:
                self.zoom_start = x
                self.zoom_box = glydget.Rectangle(x + 1, self.window.height -\
                                  self.geometry.vertical_margin - 5, 1,
                                  height - 2, [255,255,255,155,255,255,255,100,
                                           255,255,255,200,255,255,255,120])
                self.zoom_frame = glydget.Rectangle(x, self.window.height -\
                                  self.geometry.vertical_margin - 3, 1,
                                  height, (0,0,0,200), filled = False)
                self.zoom_box.build(batch = self.batch, group = self.groups[-1])
                self.zoom_frame.build(batch = self.batch, group = self.groups[-1])
                self.zoom_box_min_x = x
                self.zoom_box_max_x = x+1
                # Push the handlers for the box.
                self.zoom_handlers = \
                        self.win.window.push_handlers(on_mouse_motion, on_mouse_press)
        # Push the handlers.
        self.win.window.push_handlers(on_mouse_drag)


    def mouse_scroll(self, x, y, scroll_x, scroll_y):
        """
        Called every time the mouse wheel is scrolled.
        """
        # Check if in the menu.
        if x > self.menu_start:
            # Scroll the menu.
            self.menu.scrollMenu(scroll_y)
        # Otherwise scroll the waveforms
        else:
            self.waveform_offset += 4 * scroll_y
            if self.waveform_offset > 0:
                self.waveform_offset = 0
            # Avoid going too far down.
            max_view = self.max_viewable + self.win.geometry.time_scale
            if self.current_view_span - self.waveform_offset > max_view:
                if self.current_view_span > max_view:
                    self.waveform_offset = 0
                else:
                    self.waveform_offset = -((10 + max_view) - \
                                          self.current_view_span)
            # Update the scroll_bar.
            self.scroll_bar.changePosition()

class Geometry(object):
    """
    Stores all geometry information necessary in a central place.
    """
    def __init__(self, *args, **kwargs):
        """
        Sets default values.
        """
        # Set default margins.
        self.vertical_margin = kwargs.get('vertical_margin', 10)
        self.horizontal_margin = kwargs.get('horizontal_margin', 10)
        # Width of the scroll_bar.
        self.scroll_bar_width = kwargs.get('scroll_bar_width', 10)
        # Width of the menu
        self.menu_width = kwargs.get('menu_width', 10)
        # Height of the time_scale.
        self.time_scale = 63
        # Height of the status_bar.
        self.status_bar_height = 16
        # The offset of all actual waveform graphs on the left side.
        self.graph_start_x = 140
        # Height of a single waveform plot object. Will be updated if it
        # changes.
        self.plot_height = 80
        self.min_plot_height = 25
        self.max_plot_height = 80
        # Inner padding of the waveform plots.. Currently used for top, bottom
        # and left border.
        self.graph_pad = 2

