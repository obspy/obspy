#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from gui_element import GUIElement
import glydget
from itertools import izip
import numpy as np
from numpy.ma import is_masked
from obspy.core import read, UTCDateTime
import pyglet
from waveform_group import WaveformGroup

class WaveformPlot(GUIElement):
    """
    Handles the actual drawing of the Waveforms. One WaveformPlot object is
    needed for each seperate trace.

    Will automatically add itself to the windows waveform list and therefore be
    automatically displayed.
    """
    def __init__(self, *args, **kwargs):
        """
        Usual init method.
        """
        super(WaveformPlot, self).__init__(self, **kwargs)
        # Dummy value for header.
        self.header = ""
        # XXX: Will need to be read from the file.
        self.starttime = kwargs.get('starttime', UTCDateTime(0))
        self.endtime = kwargs.get('endtime', UTCDateTime(1971,1,1))
        self.time_range = self.endtime - self.starttime
        # Path to the files.
        self.id = kwargs.get('id', None)
        # Prepare the data.
        self.stream = kwargs.get('stream', None)
        self.readAndMergeData()
        if self.stream is None:
            msg = 'Error fetching %s.%s.%s.%s from %s' % (self.id[0],
                    self.id[1], self.id[2], self.id[3], self.win.seishub.server)
            self.win.status_bar.setError(msg)
            del self
        # Height of plot.
        self.height = self.win.geometry.plot_height
        self.max_height = kwargs.get('max_height', 80)
        self.min_height = kwargs.get('min_height', 25)
        # Inner padding. Currently used for top, bottom and left border.
        self.pad = self.win.geometry.graph_pad
        # Bind to waveform list of window.
        self.win.waveforms.append(self)
        # Current position in self.win.waveforms.
        self.position = len(self.win.waveforms)
        # Get the offsets.
        self.calculateOffsets()
        # Group to handle the positioning and the scrolling with the GPU.
        self.group1 = WaveformGroup(self.win, self.x_offset, self.y_offset, 1)
        self.group2 = WaveformGroup(self.win, self.x_offset, self.y_offset, 2)
        self.group3 = WaveformGroup(self.win, self.x_offset, self.y_offset, 3)
        self.plot_group = WaveformGroup(self.win, self.x_offset, self.y_offset, 2,
                                   plot = True)
        # Create and bind plot.
        self.initPlot()
        # Create title.
        self.createTitle()
        # Update the viewable area of the window.
        self.win.max_viewable = len(self.win.waveforms) * \
                                (self.win.geometry.vertical_margin + self.height)
        # Push the handlers to handle mouse clicks on the waveform plot.
        self.win.window.push_handlers(self.on_mouse_press)
        # Adapt the plot size upon creating a new plot.
        self.win.utils.adaptPlotHeight()

    def calculateOffsets(self):
        """
        Read self.position and write the current offsets to class variables.
        """
        self.x_offset = self.win.geometry.horizontal_margin
        #self.y_offset = (self.position)*(self.height + \
        #                self.win.geometry.vertical_margin)
        # XXX: Some fault...
        self.y_offset = (self.position - 1)*(self.height + \
                        self.win.geometry.vertical_margin) + \
                        self.win.geometry.time_scale + \
                        self.win.geometry.vertical_margin

    def getOwnGeometry(self):
        """
        Return a tuple with(x_start, x_end, y_start, y_end)
        """
        return (self.x_offset, self.x_offset + self.width,
                self.win.window.height - self.y_offset - self.height, 
                self.win.window.height - self.y_offset)

    def updatePosition(self):
        """
        Updates the position of the graph after one other graph has been
        deleted.
        """
        # Recalculate the position.
        self.calculateOffsets()
        # Update all the groups.
        for group in [self.group1, self.group2, self.group3, self.plot_group]:
            group.x_offset = self.x_offset
            group.y_offset = self.y_offset
        self.win.max_viewable = len(self.win.waveforms) * \
                                (self.win.geometry.vertical_margin + self.height)

    def updateHeight(self):
        """
        Updates the height of the waveform plot.
        """
        self.height = self.win.geometry.plot_height
        self.resize(self.win.window.width, self.win.window.height)
        self.updatePosition()

    def readAndMergeData(self):
        """
        Reads the data to self.stream
        """
        # Copy the stream to always have an original version.
        self.org_stream = deepcopy(self.stream)

    def createMinMaxList(self):
        """
        Creates the minmax list. The method used is not fully accurate but the results
        should be ok.
        """
        pixel = self.win.detail
        data = self.stream.slice(self.starttime, self.endtime)[0].data 
        # Reshape and calculate point to point differences.
        per_pixel = int(len(data)//pixel)
        ptp = data[:pixel * per_pixel].reshape(pixel, per_pixel).ptp(axis=1)
        # Last pixel.
        last_pixel = data[pixel * per_pixel:]
        if len(last_pixel):
            last_pixel = last_pixel.ptp()
            if ptp[-1] < last_pixel:
                ptp[-1] = last_pixel
        self.ptp = ptp.astype('float32')
        # Create a logarithmic axis.
        if self.win.log_scale:
            self.ptp += 1
            self.ptp = np.log(self.ptp)/np.log(self.win.log_scale)
        # Make it go from 0 to 100.
        self.ptp *= 100.0/self.ptp.max()
        # Set masked arrays to zero.
        if is_masked(self.ptp):
            self.ptp.fill_value = 0.0
            self.ptp = self.ptp.filled()
        # Assure that very small values are also visible. Only true gaps are 0
        # and will stay 0.
        self.ptp[(self.ptp > 0) & (self.ptp < 0.5)] = 0.5

    def addDeleteButton(self):
        """
        Adds a delete Button to the plot.
        """
        # Show delete Button.
        button_text = "{font_name 'Arial'}{font_size 10}" + \
                     "{color (255, 0, 0, 255)}[del]"
        button_document = pyglet.text.decode_attributed(button_text)
        self.delete_button_end_x = self.graph_start_x - 5
        self.delete_button_start_y = - 5
        self.button_layout = pyglet.text.DocumentLabel(document = button_document,
                          x = self.delete_button_end_x, y =
                              self.delete_button_start_y, width = 20,
                              batch = self.batch, anchor_x = 'right', anchor_y = 'top',
                              group = self.group1, multiline = False)

    def on_mouse_press(self, x, y, button, modifiers):
        """
        Handles all mouse presses inside the Waveform plot.
        """
        x_min, x_max, y_min, y_max = self.getOwnGeometry()
        # Check whether or not in bounds.
        if x >= x_min and x <= x_max and y >= y_min and y <= y_max:
            # Check whether or not the delete Button was hit.
            if x < x_min + self.delete_button_end_x and \
               x > x_min + self.delete_button_end_x - 22 and \
               y < y_max + self.delete_button_start_y and \
               y > y_max + self.delete_button_start_y - 15:
                # Delete.
                position = self.position
                self._delete()
                # Adjust the position of all following waveform plots.
                for waveform in self.win.waveforms:
                    if waveform.position > position:
                        waveform.position -= 1
                        waveform.updatePosition()
                # The position of the plot is also the position in the collected
                # waveforms.
                # XXX: Test if this is always correct. This is a likely error.
                del self.win.waveforms[position - 1]
                # Set new status text.
                self.win.status_bar.setText('%i Traces' % len(self.win.waveforms))

    def _delete(self):
        """
        Method the delete the object. Only takes care of the objects in the
        frame buffer. The garbage collector should be able to take care of the
        rest.
        """
        # Empty plots are possible.
        try:
            self.bars.delete()
        except:
            pass
        self.title_layout.delete()
        self.button_layout.delete()
        self.box.delete()
        self.graph_box.delete()
        self.plot.delete()

    def replot(self):
        """
        Draws the plot again with updated global informations.
        """
        try:
            self.bars.delete()
        except:
            pass
        # Look if current stream is in bounds.
        print self.win.starttime, self.stream[0].stats.starttime
        print self.win.endtime, self.stream[0].stats.endtime
        if self.win.starttime >= self.stream[0].stats.starttime and \
          self.win.endtime <= self.stream[0].stats.endtime:
            if self.env.debug:
                print 'Data already in buffer.'
            self.stream = deepcopy(self.org_stream)
        # Otherwise reload the data.
        else:
            if self.env.debug:
                print 'Requested time span not fully in buffer, requesting' +\
                      ' more.'
            s = self.stream[0].stats
            self.stream = self.win.utils.add_plot(s.network, s.station,
                                 s.location, s.channel, new_plot = False)
            self.org_stream = deepcopy(self.stream)
        self.initPlot(create = False)

    def initPlot(self, create = True):
        """
        Inits the plot.
        """
        # Trim the data. Pad if necessary.
        self.stream.trim(self.win.starttime, self.win.endtime, pad = True)
        # Writes class attributes.
        self.starttime = self.stream[0].stats.starttime
        self.endtime = self.stream[0].stats.endtime
        self.time_range = self.endtime - self.starttime
        self.header = self.stream[0].getId()
        if create:
            self.createPlot()
        # Get the point-to-point array.
        self.createMinMaxList()
        # Create an array containing the quad coordinates for each vertical
        # line.
        quads = np.empty(self.win.detail * 8)
        # Write the x_values. They will go from 0 to 100 and later be stretched
        # with OpenGL calls.
        x_values = np.linspace(0, 100, self.win.detail + 1)
        quads[0::8] = x_values[:-1]
        quads[2::8] = x_values[:-1]
        quads[4::8] = x_values[1:]
        quads[6::8] = x_values[1:]
        # Write the y-values.
        self.ptp /= 2.0
        middle = 50.0
        quads[1::8] = middle - self.ptp
        quads[3::8] = middle + self.ptp
        quads[5::8] = middle + self.ptp
        quads[7::8] = middle - self.ptp
        self.quads = quads
        color = [0, 0, 0] * (len(quads)/2)
        self.bars  = self.batch.add(len(quads)/2,
                        pyglet.gl.GL_QUADS, self.plot_group,
                        ('v2f', self.quads),
                        ('c3B', tuple(color)))
        # Set new status text.
        self.win.status_bar.setText('%i Traces' % len(self.win.waveforms))
    
    def createTitle(self):
        """
        Reads the class attributes and creates a title.
        """
        # Position.
        x =  5
        y = - 5
        # Set the title.
        title_document = pyglet.text.decode_text(self.header)
        title_document.set_style(0, 2, dict(font_name='arial',
                                 bold=True, font_size=10, color=(0,0,0,255)))
        self.title_layout = pyglet.text.DocumentLabel(document = title_document,
                          x = x , y = y, batch = self.batch, anchor_x = 'left',
                          anchor_y = 'top', group = self.group1)

    def createPlot(self):
        """
        Create and bind to batch and group. Each WaveformPlot object will have
        its own WaveformGroup for handling all the transforms needed.

        Therefore scrolling and placement of the plot will be entirely handled
        by the graphics card. Only the width and height need to be handled by
        the CPU.

        The plot will go from (0,0) to (self.win.window.width - 2 *
        self.win.geometry.horizontal_margin - self.win.geometry.scroll_bar_width, self.height).
        """
        self.width = self.win.window.width - 3 * self.win.geometry.horizontal_margin - \
                self.win.geometry.scroll_bar_width - self.win.geometry.menu_width
        # The box is slighty smaller than the border.
        self.plot = glydget.Rectangle(1, -1, self.width-2,
                        self.height-2,
                        [255, 255, 255, 250, 255, 255, 255, 240,
                        255, 255, 255, 230, 255, 255, 255, 215])
        # also create a box around it.
        self.box = glydget.Rectangle(0, 0, self.width,
                        self.height,
                        (205, 55, 55, 250), filled=False)
        # Add to batch.
        self.plot.build(batch = self.batch, group = self.group1)
        self.box.build(batch = self.batch, group = self.group1)
        # Add actual graph. Also set class variables for later easier access.
        self.graph_start_x = self.win.geometry.graph_start_x
        self.graph_start_y = -self.pad
        self.graph_width = self.width - self.graph_start_x - self.pad - 1
        # Set the scaling factor of the plot accordingly.
        self.plot_group.plot = (self.graph_width/100.0, (self.height-6.0)/100.0)
        self.plot_group.plot_offset = self.graph_start_x
        self.graph_height = self.height - 2*self.pad
        self.graph_box = glydget.Rectangle(self.graph_start_x,
                         self.graph_start_y, self.graph_width + 1,
                         self.graph_height,
                        (0, 0, 0, 250), filled=False)
        self.graph_box.build(batch = self.batch, group = self.group2)
        # Create a delete Button.
        self.addDeleteButton()
        # Add to object_list.
        self.win.object_list.append(self)

    def resize(self, width, height):
        """
        All adjustments neccessary on resize.
        """
        self.width = self.win.window.width - 3 * self.win.geometry.horizontal_margin - \
                self.win.geometry.scroll_bar_width - self.win.geometry.menu_width
        # The box is slighty smaller than the border.
        self.plot.begin_update()
        self.plot.resize(self.width - 2,self.height-2)
        self.plot.end_update()
        self.box.begin_update()
        self.box.resize(self.width,self.height)
        self.box.end_update()
        # Update the graph. Also update the variables.
        self.graph_width = self.width - self.graph_start_x - self.pad - 1
        self.graph_height = self.height - 4
        self.graph_box.begin_update()
        self.graph_box.resize(self.graph_width + 1, self.graph_height)
        self.graph_box.end_update()
        # Change the scaling factor of the plot.
        self.plot_group.plot = (self.graph_width/100.0, (self.height-6)/100.0)
        self.win.max_viewable = len(self.win.waveforms) * \
                                (self.win.geometry.vertical_margin + self.height)
