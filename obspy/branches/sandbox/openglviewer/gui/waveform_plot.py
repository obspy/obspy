#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gui_element import GUIElement
import glydget
from obspy.core import UTCDateTime
import pyglet

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
        # Change group to a Custom waveform group.
        self.group = self.win.waveform_group
        # XXX: Will need to be read from the file.
        self.starttime = kwargs.get('starttime', UTCDateTime(0))
        self.endtime = kwargs.get('endtime', UTCDateTime(3600))
        # Symetric margins around the plots.
        self.vertical_margin = kwargs.get('vertical_margin', 10)
        self.horizontal_margin = kwargs.get('horizontal_margin', 10)
        # Height of plot.
        self.height = kwargs.get('height', 200)
        # Bind to waveform list of window.
        self.win.waveforms.append(self)
        # Current position in self.win.waveforms.
        self.position = len(self.win.waveforms)
        # Create and bind plot.
        self.createPlot()
        # Create title.
        self.createTitle()
        # Update the viewable area of the window.
        self.win.max_viewable = len(self.win.waveforms) * \
                                (self.vertical_margin + self.height)

    def createTitle(self):
        """
        XXX: Dummy title to ease development.
        """
        # Position.
        x = self.horizontal_margin + 5
        y = self._get_y() - 5
        # Just write the position to it.
        title_document = pyglet.text.decode_text(str(self.position))
        title_document.set_style(0, 2, dict(font_name='arial',
                                 bold=True, font_size=10, color=(0,0,0,255)))
        self.title_layout = pyglet.text.DocumentLabel(document = title_document,
                          x = x , y = y, batch = self.batch, anchor_x = 'left',
                          anchor_y = 'top', group = self.group)

    def _get_y(self):
        """
        Helper method to get the y_value of the top left corner.
        """
        y = self.win.window.height - ((self.position - 1) * self.height +\
            self.position * self.vertical_margin)
        return y

    def createPlot(self):
        """
        Create and bind to batch.
        """
        y = self._get_y()
        width = self.win.window.width - 2 * self.horizontal_margin - \
                self.win.scroll_bar.width
        # The box is slighty smaller than the border.
        self.plot = glydget.Rectangle(self.horizontal_margin+1, y-1, width-2,
                        self.height-2,
                        [255, 255, 255, 250, 255, 255, 255, 240,
                        255, 255, 255, 230, 255, 255, 255, 215])
        # Also create a box around it.
        self.box = glydget.Rectangle(self.horizontal_margin, y, width,
                        self.height,
                        (205, 55, 55, 250), filled=False)
        # Add to batch.
        self.plot.build(batch = self.batch, group = self.group)
        self.box.build(batch = self.batch, group = self.group)
        # Add to object_list.
        self.win.object_list.append(self)

    def resize(self, width, height):
        """
        All adjustments neccessary on resize.
        """
        width = self.win.window.width - 2 * self.horizontal_margin -\
                self.win.scroll_bar.width
        y = self._get_y()
        # The box is slighty smaller than the border.
        self.plot.begin_update()
        self.plot.move(self.horizontal_margin+1,y-1)
        self.plot.resize(width-2,self.height-2)
        self.plot.end_update()
        self.box.begin_update()
        self.box.move(self.horizontal_margin,y)
        self.box.resize(width,self.height)
        self.box.end_update()
        # Update text.
        self.title_layout.begin_update()
        self.title_layout.y = y - 5
        self.title_layout.end_update()
