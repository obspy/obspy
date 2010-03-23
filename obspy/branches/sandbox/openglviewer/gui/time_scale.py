#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gui_element import GUIElement
from obspy.core import UTCDateTime
import theme
import glydget
import pyglet

class TimeScale(GUIElement):
    """
    Creates and updates the time scale.
    """
    def __init__(self, *args, **kwargs):
        super(TimeScale, self).__init__(self, **kwargs)
        # Items that collect all subitems for easy tracking and deleting.
        self.year_boxes = []
        self.year_labels = []
        self.month_items = []
        self.day_items = []
        # Create the time scale.
        self.createTimeScale()
        # Create the subscales.
        self.yearScale()

    def createTimeScale(self):
        """
        Actually creates the time_scale.
        """
        self.width = self.win.window.width - 3 * \
                self.win.geometry.horizontal_margin - \
                self.win.geometry.scroll_bar_width - \
                self.win.geometry.menu_width
        geo = self.win.geometry
        # Make thinner for the borders.
        self.top_box = glydget.Rectangle(geo.horizontal_margin + 2,
                        self.win.window.height - geo.vertical_margin - 2,
                        self.width - 4,
                        geo.time_scale - geo.vertical_margin - 4, 
                        (255, 255, 255, 255))
        self.bottom_box = glydget.Rectangle(geo.horizontal_margin + 2,
                        geo.status_bar_height + geo.time_scale - 2,
                        self.width - 4,
                        geo.time_scale - geo.vertical_margin - 4, 
                        (255, 255, 255, 255))
        # Add two frames for a thick border.
        self.top_frame1 = glydget.Rectangle(geo.horizontal_margin,
                        self.win.window.height - geo.vertical_margin,
                        self.width,
                        geo.time_scale - geo.vertical_margin,
                        (0, 0, 0, 255), filled = False)
        self.bottom_frame1 = glydget.Rectangle(geo.horizontal_margin,
                        geo.status_bar_height + geo.time_scale, self.width,
                        geo.time_scale - geo.vertical_margin,
                        (0, 0, 0, 255), filled = False)
        self.top_frame2 = glydget.Rectangle(geo.horizontal_margin + 1,
                        self.win.window.height - geo.vertical_margin - 1,
                        self.width - 2,
                        geo.time_scale - geo.vertical_margin - 2,
                        (0, 0, 0, 255), filled = False)
        self.bottom_frame2 = glydget.Rectangle(geo.horizontal_margin + 1,
                        geo.status_bar_height + geo.time_scale - 1,
                        self.width - 2,
                        geo.time_scale - geo.vertical_margin - 2,
                        (0, 0, 0, 255), filled = False)
        # Add every Rectangle to the batch of the window.
        self.top_box.build(batch = self.batch, group = self.group)
        self.bottom_box.build(batch = self.batch, group = self.group)
        self.top_frame1.build(batch = self.batch, group = self.group)
        self.bottom_frame1.build(batch = self.batch, group = self.group)
        self.top_frame2.build(batch = self.batch, group = self.group)
        self.bottom_frame2.build(batch = self.batch, group = self.group)
        # Add boxes for year, month, day.
        self.year_frame_t = glydget.Rectangle(\
                            geo.horizontal_margin + geo.graph_start_x,
                            self.win.window.height - geo.vertical_margin - 3,
                            self.width - geo.graph_start_x - 3, 10,
                            (100, 100, 100, 255), filled = False)
        self.month_frame_t = glydget.Rectangle(\
                            geo.horizontal_margin + geo.graph_start_x,
                            self.win.window.height - geo.vertical_margin - 14,
                            self.width - geo.graph_start_x - 3, 10,
                            (100, 100, 100, 255), filled = False)
        self.day_frame_t = glydget.Rectangle(\
                            geo.horizontal_margin + geo.graph_start_x,
                            self.win.window.height - geo.vertical_margin - 25,
                            self.width - geo.graph_start_x - 3, 10,
                            (100, 100, 100, 255), filled = False)
        self.year_frame_b = glydget.Rectangle(\
                            geo.horizontal_margin + geo.graph_start_x,
                            geo.status_bar_height + geo.time_scale - 3,
                            self.width - geo.graph_start_x - 3, 10,
                            (100, 100, 100, 255), filled = False)
        self.month_frame_b = glydget.Rectangle(\
                            geo.horizontal_margin + geo.graph_start_x,
                            geo.status_bar_height + geo.time_scale - 14,
                            self.width - geo.graph_start_x - 3, 10,
                            (100, 100, 100, 255), filled = False)
        self.day_frame_b = glydget.Rectangle(\
                            geo.horizontal_margin + geo.graph_start_x,
                            geo.status_bar_height + geo.time_scale - 25,
                            self.width - geo.graph_start_x - 3, 10,
                            (100, 100, 100, 255), filled = False)
        # Add boxes to batch.
        self.year_frame_t.build(batch = self.batch, group = self.group)
        self.month_frame_t.build(batch = self.batch, group = self.group)
        self.day_frame_t.build(batch = self.batch, group = self.group)
        self.year_frame_b.build(batch = self.batch, group = self.group)
        self.month_frame_b.build(batch = self.batch, group = self.group)
        self.day_frame_b.build(batch = self.batch, group = self.group)
        # Add to object_list.
        self.win.object_list.append(self)
        
    def yearScale(self, top = True):
        """
        Creates the subdivisions of the year scale.
        """
        # Shortcut to window geometry.
        geo = self.win.geometry
        starttime = self.win.starttime
        endtime = self.win.endtime
        time_range = float(endtime - starttime)
        # Pixel counts.
        start_x = geo.horizontal_margin + geo.graph_start_x + 1
        end_x =  start_x + self.width - geo.graph_start_x - 4
        x_range = end_x - start_x
        # Top or bottom year scala.
        if top:
            start_y = self.win.window.height - geo.vertical_margin - 4
        else:
            start_y = geo.status_bar_height + geo.time_scale - 4
        end_y = start_y - 8
        y_range = 8
        # Get the number of years.
        year_count = endtime.year - starttime.year
        years = range(starttime.year, endtime.year + 1)
        # Loop over all years.
        even_color = (200,200,200,255)
        odd_color = (150,150,150,255)
        for _i, year in enumerate(years):
            # Some variables.
            start_of_year = UTCDateTime(year, 1, 1)
            end_of_year = UTCDateTime(year+1, 1, 1) - 1
            # Calculate boundaries.
            start_frac = (start_of_year - starttime) / time_range
            if start_frac < 0:
                start_frac = 0
            start = start_frac * x_range
            if start < 0:
                start = 0
            start += start_x
            end_frac = (endtime - end_of_year) / time_range
            end_frac = 1.0 - end_frac
            if end_frac > 1.0:
                end_frac = 1.0
            end = end_frac * x_range
            end = x_range - end
            if end > x_range:
                end = x_range
            end += start_x
            graph_width = (end_frac - start_frac) * x_range
            if _i%2:
                color = odd_color
            else:
                color = even_color
            # Add half a pixel to avoid rounding issues.
            year_box = glydget.Rectangle(start, start_y,
                        graph_width + 0.5, y_range, color)
            year_box.build(batch = self.batch, group = self.group)
            # Add to list for easier tracking.
            self.year_boxes.append((year_box, start_frac, end_frac))
            # If two narrow do not add a name.
            if graph_width < 30:
                continue
            # Add name.
            year_document = pyglet.text.decode_text(str(year))
            year_document.set_style(0, 5, dict(font_name='Arial', font_size=8.5, 
                                          color = (0,0,0,255), bold = True))
            self.year_layout = pyglet.text.DocumentLabel(document = year_document,
                               x = (end_frac + start_frac)/2 * x_range + start_x,
                               y = start_y - 11.0,
                               batch = self.batch, anchor_x = 'center',
                               anchor_y = 'bottom', group = self.group)
            self.year_labels.append((self.year_layout, start_frac, end_frac))

    def monthScale(self, top = True):
        """
        Creates the subdivisions of the month scale.
        """
        # Shortcut to window geometry.
        geo = self.win.geometry
        starttime = self.win.starttime
        endtime = self.win.endtime
        time_range = float(endtime - starttime)
        # Pixel counts.
        start_x = geo.horizontal_margin + geo.graph_start_x + 1
        end_x =  start_x + self.width - geo.graph_start_x - 4
        x_range = end_x - start_x
        # Top or bottom year scala.
        if top:
            start_y = self.win.window.height - geo.vertical_margin - 4
        else:
            start_y = geo.status_bar_height + geo.time_scale - 4
        end_y = start_y - 8
        y_range = 8
        # Get the number of months.
        
        years = range(starttime.year, endtime.year + 1)
        # Loop over all years.
        even_color = (200,200,200,255)
        odd_color = (150,150,150,255)
        for _i, year in enumerate(years):
            # Some variables.
            start_of_year = UTCDateTime(year, 1, 1)
            end_of_year = UTCDateTime(year+1, 1, 1) - 1
            # Calculate boundaries.
            start_frac = (start_of_year - starttime) / time_range
            if start_frac < 0:
                start_frac = 0
            start = start_frac * x_range
            if start < 0:
                start = 0
            start += start_x
            end_frac = (endtime - end_of_year) / time_range
            end_frac = 1.0 - end_frac
            if end_frac > 1.0:
                end_frac = 1.0
            end = end_frac * x_range
            end = x_range - end
            if end > x_range:
                end = x_range
            end += start_x
            graph_width = (end_frac - start_frac) * x_range
            if _i%2:
                color = odd_color
            else:
                color = even_color
            # Add half a pixel to avoid rounding issues.
            year_box = glydget.Rectangle(start, start_y,
                        graph_width + 0.5, y_range, color)
            year_box.build(batch = self.batch, group = self.group)
            # Add to list for easier tracking.
            self.year_boxes.append((year_box, start_frac, end_frac))
            # If two narrow do not add a name.
            if graph_width < 30:
                continue
            # Add name.
            year_document = pyglet.text.decode_text(str(year))
            year_document.set_style(0, 5, dict(font_name='Arial', font_size=8.5, 
                                          color = (0,0,0,255), bold = True))
            self.year_layout = pyglet.text.DocumentLabel(document = year_document,
                               x = (end_frac + start_frac)/2 * x_range + start_x,
                               y = start_y - 11.0,
                               batch = self.batch, anchor_x = 'center',
                               anchor_y = 'bottom', group = self.group)
            self.year_labels.append((self.year_layout, start_frac, end_frac))

    def resizeYearTimeScale(self, width, height, top = True):
        """
        Resizes the year subscale.
        """
        # Shortcut to window geometry.
        geo = self.win.geometry
        # Pixel counts.
        start_x = geo.horizontal_margin + geo.graph_start_x + 1
        end_x =  start_x + self.width - geo.graph_start_x - 4
        x_range = float(end_x - start_x)
        # Top or bottom year scala.
        if top:
            start_y = self.win.window.height - geo.vertical_margin - 4
        else:
            start_y = geo.status_bar_height + geo.time_scale - 4
        end_y = start_y - 8
        y_range = 8
        # Resize and move the boxes.
        for year in self.year_boxes:
            year[0].begin_update()
            # Add half a pixel to avoid rounding issues.
            year[0].resize((year[2] - year[1]) * x_range + 0.5, 8)
            year[0].move(year[1] * x_range + start_x, start_y)
            year[0].end_update()
        # Move the labels.
        for label in self.year_labels:
            print label[1], label[2]
            label[0].begin_update()
            label[0].x = (label[2] + label[1])/2 * x_range + start_x
            label[0].y = start_y - 11
            label[0].end_update()

    def changeTimeScale(self):
        """
        Changes the time scale.
        """
        # Delete all old items.
        for year in self.year_boxes:
            year[0].delete()
        for label in self.year_labels:
            label[0].delete()
        self.year_boxes = []
        self.year_labels = []
        # Create new time subscales.
        self.yearScale()

    def resize(self, width, height):
        """
        Handles the resizing operations.
        """
        # Define new width.
        self.width = width - 3 * \
                self.win.geometry.horizontal_margin - \
                self.win.geometry.scroll_bar_width - \
                self.win.geometry.menu_width
        # Shortcut to the window geometry.
        geo = self.win.geometry
        # Handle the resizing of the boxes.
        self.top_box.begin_update()
        self.top_box.move(geo.horizontal_margin + 2,height -\
                          geo.vertical_margin - 2)
        self.top_box.resize(self.width - 4, geo.time_scale - \
                            geo.vertical_margin - 4)
        self.top_box.end_update()
        self.bottom_box.begin_update()
        self.bottom_box.resize(self.width - 4, geo.time_scale - \
                               geo.vertical_margin - 4)
        self.bottom_box.end_update()
        # Handle the resizing of the frames.
        self.top_frame1.begin_update()
        self.top_frame1.move(geo.horizontal_margin, height -\
                          geo.vertical_margin)
        self.top_frame1.resize(self.width, geo.time_scale - \
                            geo.vertical_margin)
        self.top_frame1.end_update()
        self.top_frame2.begin_update()
        self.top_frame2.move(geo.horizontal_margin + 1, height -\
                          geo.vertical_margin - 1)
        self.top_frame2.resize(self.width - 2, geo.time_scale - \
                            geo.vertical_margin - 2)
        self.top_frame2.end_update()
        self.bottom_frame1.begin_update()
        self.bottom_frame1.resize(self.width, geo.time_scale - \
                               geo.vertical_margin)
        self.bottom_frame1.end_update()
        self.bottom_frame2.begin_update()
        self.bottom_frame2.resize(self.width - 2, geo.time_scale - \
                               geo.vertical_margin - 2)
        self.bottom_frame2.end_update()
        # Resize the time scales.
        self.year_frame_t.begin_update()
        self.year_frame_t.resize(self.width - geo.graph_start_x - 3, 10)
        self.year_frame_t.move(geo.horizontal_margin + geo.graph_start_x,
                            self.win.window.height - geo.vertical_margin - 3)
        self.year_frame_t.end_update()

        self.month_frame_t.begin_update()
        self.month_frame_t.resize(self.width - geo.graph_start_x - 3, 10)
        self.month_frame_t.move(geo.horizontal_margin + geo.graph_start_x,
                            self.win.window.height - geo.vertical_margin - 14)
        self.month_frame_t.end_update()

        self.day_frame_t.begin_update()
        self.day_frame_t.resize(self.width - geo.graph_start_x - 3, 10)
        self.day_frame_t.move(geo.horizontal_margin + geo.graph_start_x,
                            self.win.window.height - geo.vertical_margin - 25)
        self.day_frame_t.end_update()

        self.year_frame_b.begin_update()
        self.year_frame_b.resize(self.width - geo.graph_start_x - 3, 10)
        self.year_frame_b.end_update()

        self.month_frame_b.begin_update()
        self.month_frame_b.resize(self.width - geo.graph_start_x - 3, 10)
        self.month_frame_b.end_update()

        self.day_frame_b.begin_update()
        self.day_frame_b.resize(self.width - geo.graph_start_x - 3, 10)
        self.day_frame_b.end_update()

        # Resize the year subscale.
        self.resizeYearTimeScale(width, height)
