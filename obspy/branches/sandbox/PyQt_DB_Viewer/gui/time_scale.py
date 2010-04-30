#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt4 import QtCore, QtGui, QtOpenGL

from obspy.core import UTCDateTime
from utils import MONTHS

class TimeScale(QtGui.QGraphicsItemGroup):
    """
    Creates and handles the time scale.
    """
    def __init__(self, env, parent, width, *args, **kwargs):
        super(TimeScale, self).__init__()
        # Set the parent.
        self.env = env
        self.width = width

        # Some default values.
        self.subscale_height = 15

        # Define some colors.
        self.color1 = QtGui.QColor(200,200,200,255)
        self.color2 = QtGui.QColor(120,120,120,255)
        self.bg_color = self.color1

        # Lists that collect all subitems for easy tracking and deleting.
        self.year_boxes = []
        self.year_labels = []
        self.month_boxes = []
        self.month_labels = []
        self.day_boxes = []
        self.day_labels = []
        # Create the time scale.

        self.createTimeScale()
        # Create the subscales.
        self.yearScale()
        self.monthScale()
        self.dayScale()

    def createTimeScale(self):
        """
        Actually creates the time_scale.
        """
        # Create the background for the whole time_scale.
        self.background = QtGui.QGraphicsRectItem(-2000, 0, 4000, 55)
        self.background.setBrush(self.bg_color)
        self.background.setZValue(-200)
        self.addToGroup(self.background)
        # Create the backgrounds for each subscale. The month just takes the
        # normal background color.
        self.year_bg = QtGui.QGraphicsRectItem(-2000,0,4000,15)
        self.year_bg.setBrush(self.color2)
        self.year_bg.setZValue(-199)
        self.addToGroup(self.year_bg)
        self.day_bg = QtGui.QGraphicsRectItem(-2000,30,4000,15)
        self.day_bg.setBrush(self.color2)
        self.day_bg.setZValue(-199)
        self.addToGroup(self.day_bg)
        
    def yearScale(self, top = True):
        """
        Creates the subdivisions of the year scale.
        """
        color = self.color1
        # Shortcut to window geometry.
        starttime = self.env.starttime
        endtime = self.env.endtime
        time_range = float(endtime - starttime)
        # Pixel counts.
        start_x = 0
        end_x =  self.width - start_x
        print 'self.width:', self.width
        x_range = end_x - start_x
        start_y = 0
        end_y = self.subscale_height
        y_range = self.subscale_height
        # Get the number of years.
        year_count = endtime.year - starttime.year
        print year_count
        years = range(starttime.year, endtime.year + 1)
        print years
        # Loop over all years.
        for _i, year in enumerate(years):
            print year
            # Some variables.
            start_of_year = UTCDateTime(year, 1, 1)
            end_of_year = UTCDateTime(year+1, 1, 1)
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
            # Only draw every second box.
            if _i%2:
                year_box = QtGui.QGraphicsRectItem(start, start_y,
                            graph_width, y_range)
                year_box.setBrush(color)
                year_box.setZValue(-198)
                self.addToGroup(year_box)
                # Add to list for easier tracking.
                self.year_boxes.append((year_box, start_frac, end_frac))
            # If two narrow do not add a name.
            if graph_width < 30:
                continue
            # Add name.
            year_name = QtGui.QGraphicsSimpleTextItem(str(year))
            rect = year_name.boundingRect()
            # XXX: The +2 at the end is just trial end error. I cannot figure
            # out a way to calculate it. The height of the rectangle is 16
            # while the year subscale height is only 15. But the text is still
            # positioned too high without the +2.
            year_name.moveBy(((end_frac-start_frac)/2+start_frac)*self.width-\
                            rect.width()/2, start_y + 2)
            self.addToGroup(year_name)

    def monthScale(self, top = True):
        """
        Creates the subdivisions of the month scale.
        """
        color = self.color2
        # Shortcut to window geometry.
        starttime = self.env.starttime
        endtime = self.env.endtime
        time_range = float(endtime - starttime)
        # Pixel counts.
        start_x = 0
        end_x =  self.width
        x_range = end_x - start_x
        start_y = 15
        end_y = start_y + self.subscale_height
        y_range = self.subscale_height
        # Get the number of months.
        if endtime.year == starttime.year:
            months = (endtime.month - starttime.month) + 1
        else:
            months = 0
            years = endtime.year - starttime.year
            # If more than one year add twelve months per year.
            if years > 1:
                months += 12 * (years -1)
            # Get boundaries.
            months += (12 - starttime.month) + 1
            months += endtime.month
        months_count = range(months)
        # Loop over all years.
        # Loop over every month.
        for month in months_count:
            # Get the year and month of the currently treated month.
            cur_month = starttime.month + month
            if cur_month > 12:
                cur_year = starttime.year + ((cur_month -1)//12)
            else:
                cur_year = starttime.year
            cur_month = cur_month % 12
            # Account for weird modulo operation.
            if cur_month == 0:
                cur_month = 12
            # Some variables.
            start_of_month = UTCDateTime(cur_year, cur_month, 1)
            if cur_month + 1> 12:
                cur_month = 1
                cur_year += 1
            else:
                cur_month += 1
            end_of_month = UTCDateTime(cur_year, cur_month, 1)
            # Calculate boundaries.
            start_frac = (start_of_month - starttime) / time_range
            if start_frac < 0:
                start_frac = 0
            start = start_frac * x_range
            if start < 0:
                start = 0
            start += start_x
            end_frac = (endtime - end_of_month) / time_range
            end_frac = 1.0 - end_frac
            if end_frac > 1.0:
                end_frac = 1.0
            end = end_frac * x_range
            end = x_range - end
            if end > x_range:
                end = x_range
            end += start_x
            graph_width = (end_frac - start_frac) * x_range
            # Only draw every second box.
            if not month%2:
                month_box = QtGui.QGraphicsRectItem(start, start_y,
                            graph_width, y_range)
                month_box.setBrush(color)
                month_box.setZValue(-198)
                self.addToGroup(month_box)
                # Add to list for easier tracking.
                self.month_boxes.append((month_box, start_frac, end_frac))
            # If two narrow do not add a name. This has to be set once and stay
            # valid for all following months, otherwise only the names of the
            # long months might appear.
            # XXX: This might result in only the larger months' labels being
            # drawn.
            if graph_width < 30:
                continue
            # Add name.
            month = start_of_month.month
            name = MONTHS[month]
            month_name = QtGui.QGraphicsSimpleTextItem(name)
            rect = month_name.boundingRect()
            # XXX: The +2 at the end is just trial end error. I cannot figure
            # out a way to calculate it. The height of the rectangle is 16
            # while the year subscale height is only 15. But the text is still
            # positioned too high without the +2.
            month_name.moveBy(((end_frac-start_frac)/2+start_frac)*self.width-\
                            rect.width()/2, start_y + 2)
            self.addToGroup(month_name)

    def dayScale(self, top = True):
        """
        Creates the subdivisions of the month scale.
        """
        color = self.color1
        # Shortcut to window geometry.
        starttime = self.env.starttime
        endtime = self.env.endtime
        time_range = float(endtime - starttime)
        # Pixel counts.
        start_x = 0
        end_x =  self.width
        x_range = end_x - start_x
        # Top or bottom year scala.
        start_y = 30
        end_y = 45
        y_range = self.subscale_height
        # Get the number of months.
        days = int(round((endtime - starttime)/86400))
        days_count = range(days)
        # Use the middle of the starting date to later calculate the current
        # day and account for leap seconds.
        noon_of_start = UTCDateTime(starttime.year, starttime.month, 
                        starttime.day, 12, 0, 0)
        # Only draw if there are less or equal than 150 days.
        if days > 150:
            return
        # Loop over every day.
        for day in days_count:
            today = noon_of_start + 86400 * day
            # Some variables.
            start_of_day = UTCDateTime(today.year, today.month, today.day, 0, 0, 0)
            end_of_day = start_of_day + 86400
            # Calculate boundaries.
            start_frac = (start_of_day - starttime) / time_range
            if start_frac < 0:
                start_frac = 0
            start = start_frac * x_range
            if start < 0:
                start = 0
            start += start_x
            end_frac = (endtime - end_of_day) / time_range
            end_frac = 1.0 - end_frac
            if end_frac > 1.0:
                end_frac = 1.0
            end = end_frac * x_range
            end = x_range - end
            if end > x_range:
                end = x_range
            end += start_x
            graph_width = (end_frac - start_frac) * x_range
            # Only draw every second box.
            if day%2:
                day_box = QtGui.QGraphicsRectItem(start, start_y,
                            graph_width, y_range)
                day_box.setBrush(color)
                day_box.setZValue(-198)
                self.addToGroup(day_box)
                # Add to list for easier tracking.
                self.day_boxes.append((day_box, start_frac, end_frac))
            # If two narrow do not add a name.
            if graph_width < 20:
                continue
            # Add name.
            name = str(today.day)
            day_name = QtGui.QGraphicsSimpleTextItem(name)
            rect = day_name.boundingRect()
            # XXX: The +2 at the end is just trial end error. I cannot figure
            # out a way to calculate it. The height of the rectangle is 16
            # while the year subscale height is only 15. But the text is still
            # positioned too high without the +2.
            day_name.moveBy(((end_frac-start_frac)/2+start_frac)*self.width-\
                            rect.width()/2, start_y + 2)
            self.addToGroup(day_name)

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
        end_y = start_y - self.subscale_height - 2
        y_range = self.subscale_height - 2
        # Resize the background.
        self.year_bg.begin_update()
        self.year_bg.resize(x_range, y_range)
        self.year_bg.move(start_x, start_y)
        self.year_bg.end_update()

        # Resize and move the boxes.
        for year in self.year_boxes:
            year[0].begin_update()
            # Add half a pixel to avoid rounding issues.
            year[0].resize((year[2] - year[1]) * x_range, self.subscale_height - 2)
            year[0].move(year[1] * x_range + start_x, start_y)
            year[0].end_update()
        # Move the labels.
        for label in self.year_labels:
            label[0].begin_update()
            label[0].x = (label[2] + label[1])/2 * x_range + start_x
            label[0].y = start_y - 13.0
            label[0].end_update()

    def resizeMonthTimeScale(self, width, height, top = True):
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
            start_y = self.win.window.height - geo.vertical_margin -\
            self.subscale_height - 5
        else:
            start_y = geo.status_bar_height + geo.time_scale - \
            self.subscale_height - 5
        end_y = start_y - self.subscale_height - 2
        y_range = self.subscale_height - 2
        # Resize the background.
        self.month_bg.begin_update()
        self.month_bg.resize(x_range, y_range)
        self.month_bg.move(start_x, start_y)
        self.month_bg.end_update()
        # Resize and move the boxes.
        for month in self.month_boxes:
            month[0].begin_update()
            # Add half a pixel to avoid rounding issues.
            month[0].resize((month[2] - month[1]) * x_range, self.subscale_height - 2)
            month[0].move(month[1] * x_range + start_x, start_y)
            month[0].end_update()
        # Move the labels.
        for label in self.month_labels:
            label[0].begin_update()
            label[0].x = (label[2] + label[1])/2 * x_range + start_x
            label[0].y = start_y - 13.0
            label[0].end_update()


    def resizeDayTimeScale(self, width, height, top = True):
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
            start_y = self.win.window.height - geo.vertical_margin -\
            2 * self.subscale_height - 6
        else:
            start_y = geo.status_bar_height + geo.time_scale - \
            2 * self.subscale_height - 6
        end_y = start_y - self.subscale_height - 2
        y_range = self.subscale_height - 2
        # Resize the background.
        self.day_bg.begin_update()
        self.day_bg.resize(x_range, y_range)
        self.day_bg.move(start_x, start_y)
        self.day_bg.end_update()
        # Resize and move the boxes.
        for day in self.day_boxes:
            day[0].begin_update()
            # Add half a pixel to avoid rounding issues.
            day[0].resize((day[2] - day[1]) * x_range, self.subscale_height - 2)
            day[0].move(day[1] * x_range + start_x, start_y)
            day[0].end_update()
        # Move the labels.
        for label in self.day_labels:
            label[0].begin_update()
            label[0].x = (label[2] + label[1])/2 * x_range + start_x - 1
            label[0].y = start_y - 13.0
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
        for month in self.month_boxes:
            month[0].delete()
        for label in self.month_labels:
            label[0].delete()
        self.month_boxes = []
        self.month_labels = []
        for day in self.day_boxes:
            day[0].delete()
        for label in self.day_labels:
            label[0].delete()
        self.day_boxes = []
        self.day_labels = []
        # Also delete the background objects.
        # XXX: Implement differently.
        self.year_bg.delete()
        self.month_bg.delete()
        self.day_bg.delete()
        # Create new time subscales.
        self.yearScale()
        self.monthScale()
        self.dayScale()

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
        #self.bottom_box.begin_update()
        #self.bottom_box.resize(self.width - 4, geo.time_scale - \
        #                       geo.vertical_margin - 4)
        #self.bottom_box.end_update()
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
        #self.bottom_frame1.begin_update()
        #self.bottom_frame1.resize(self.width, geo.time_scale - \
        #                       geo.vertical_margin)
        #self.bottom_frame1.end_update()
        #self.bottom_frame2.begin_update()
        #self.bottom_frame2.resize(self.width - 2, geo.time_scale - \
        #                       geo.vertical_margin - 2)
        #self.bottom_frame2.end_update()
        # Resize the time scales.
        self.year_frame_t.begin_update()
        self.year_frame_t.resize(self.width - geo.graph_start_x - 3, self.subscale_height)
        self.year_frame_t.move(geo.horizontal_margin + geo.graph_start_x,
                            self.win.window.height - geo.vertical_margin - 3)
        self.year_frame_t.end_update()

        self.month_frame_t.begin_update()
        self.month_frame_t.resize(self.width - geo.graph_start_x - 3, self.subscale_height)
        self.month_frame_t.move(geo.horizontal_margin + geo.graph_start_x,
                            self.win.window.height - geo.vertical_margin -\
                                self.subscale_height - 4)
        self.month_frame_t.end_update()

        self.day_frame_t.begin_update()
        self.day_frame_t.resize(self.width - geo.graph_start_x - 3, self.subscale_height)
        self.day_frame_t.move(geo.horizontal_margin + geo.graph_start_x,
                            self.win.window.height - geo.vertical_margin -\
                            2*self.subscale_height - 5)
        self.day_frame_t.end_update()
        
        #self.year_frame_b.begin_update()
        #self.year_frame_b.resize(self.width - geo.graph_start_x - 3, self.subscale_height)
        #self.year_frame_b.end_update()

        #self.month_frame_b.begin_update()
        #self.month_frame_b.resize(self.width - geo.graph_start_x - 3, self.subscale_height)
        #self.month_frame_b.end_update()

#        self.day_frame_b.begin_update()
#        self.day_frame_b.resize(self.width - geo.graph_start_x - 3, self.subscale_height)
#        self.day_frame_b.end_update()

        # Resize the year subscale.
        self.resizeYearTimeScale(width, height)
        self.resizeMonthTimeScale(width, height)
        self.resizeDayTimeScale(width, height)
