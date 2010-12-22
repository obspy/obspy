# -*- coding: utf-8 -*-

from PyQt4 import QtGui

from obspy.core import UTCDateTime
from utils import MONTHS


class TimeScale(QtGui.QGraphicsItemGroup):
    """
    Creates and handles the time scale.
    """
    def __init__(self, env, parent, width, *args, **kwargs):
        QtGui.QGraphicsItemGroup.__init__(self)
        #super(TimeScale, self).__init__()
        # Set the parent.
        self.env = env
        self.width = width

        # Some default values.
        self.subscale_height = 15

        # Define some colors.
        self.color1 = QtGui.QColor(200, 200, 200, 255)
        self.color2 = QtGui.QColor(170, 170, 170, 255)
        self.bg_color = QtGui.QColor(200, 220, 200, 255)
        self.event_color_automatic = QtGui.QColor(255, 0, 0, 155)
        self.event_color_manual = QtGui.QColor(0, 255, 0, 155)
        self.event_color_automatic_dark = QtGui.QColor(90, 0, 0, 255)
        self.event_color_manual_dark = QtGui.QColor(0, 90, 0, 255)
        self.min_event_size = 5
        # Magnitude 5 quakes will have this size. Everythin bigger will be even
        # bigger.
        self.max_event_size = 15
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
        self.plotEvents()

    def resize(self, width, height):
        """
        Handles the resizing of the time_scale and event indicator.
        """
        self.width = width
        # Resize the events.
        for event in self.events:
            x = event.x()
            y = event.y()
            rect = event.rect()
            event.setRect(event.fraction * self.width - event.event_size / 2, rect.y(), rect.width(), rect.height())
        # Resize the scales.
        for box in self.year_boxes + self.month_boxes + self.day_boxes:
            rect = box.rect()
            start = box.start_frac * self.width
            end = box.end_frac * self.width
            box.setRect(start, rect.y(), end - start, rect.height())
        # And move the labels.
        for label in self.year_labels + self.month_labels + self.day_labels:
            rect = label.boundingRect()
            label.setX((label.start_frac + label.end_frac) / 2.0 * self.width - \
                       rect.width() / 2.0)

    def createTimeScale(self):
        """
        Actually creates the time_scale.
        """
        # Create the background for the whole time_scale.
        self.background = QtGui.QGraphicsRectItem(-2000, 0, 4000, 60)
        self.background.setBrush(self.bg_color)
        self.background.setZValue(-200)
        self.addToGroup(self.background)
        # Create the backgrounds for each subscale. The month just takes the
        # normal background color.
        self.year_bg = QtGui.QGraphicsRectItem(-2000, 0, 4000, 15)
        self.year_bg.setBrush(self.color2)
        self.year_bg.setZValue(-199)
        self.addToGroup(self.year_bg)
        self.month_bg = QtGui.QGraphicsRectItem(-2000, 15, 4000, 15)
        self.month_bg.setBrush(self.color1)
        self.month_bg.setZValue(-199)
        self.addToGroup(self.month_bg)
        self.day_bg = QtGui.QGraphicsRectItem(-2000, 30, 4000, 15)
        self.day_bg.setBrush(self.color2)
        self.day_bg.setZValue(-199)
        self.addToGroup(self.day_bg)

    def yearScale(self, top=True):
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
        end_x = self.width - start_x
        x_range = end_x - start_x
        start_y = 0
        end_y = self.subscale_height
        y_range = self.subscale_height
        # Get the number of years.
        year_count = endtime.year - starttime.year
        years = range(starttime.year, endtime.year + 1)
        # Loop over all years.
        for _i, year in enumerate(years):
            # Some variables.
            start_of_year = UTCDateTime(year, 1, 1)
            end_of_year = UTCDateTime(year + 1, 1, 1)
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
            if _i % 2:
                year_box = QtGui.QGraphicsRectItem(start, start_y,
                            graph_width, y_range)
                year_box.start_frac = start_frac
                year_box.end_frac = end_frac
                year_box.setBrush(color)
                year_box.setZValue(-198)
                self.addToGroup(year_box)
                # Add to list for easier tracking.
                self.year_boxes.append(year_box)
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
            year_name.moveBy(((end_frac - start_frac) / 2 + start_frac) * self.width - \
                            rect.width() / 2, start_y + 2)
            year_name.start_frac = start_frac
            year_name.end_frac = end_frac
            self.year_labels.append(year_name)
            self.addToGroup(year_name)

    def monthScale(self, top=True):
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
        end_x = self.width
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
                months += 12 * (years - 1)
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
                cur_year = starttime.year + ((cur_month - 1) // 12)
            else:
                cur_year = starttime.year
            cur_month = cur_month % 12
            # Account for weird modulo operation.
            if cur_month == 0:
                cur_month = 12
            # Some variables.
            start_of_month = UTCDateTime(cur_year, cur_month, 1)
            if cur_month + 1 > 12:
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
            if not month % 2:
                month_box = QtGui.QGraphicsRectItem(start, start_y,
                            graph_width, y_range)
                month_box.start_frac = start_frac
                month_box.end_frac = end_frac
                month_box.setBrush(color)
                month_box.setZValue(-198)
                self.addToGroup(month_box)
                # Add to list for easier tracking.
                self.month_boxes.append(month_box)
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
            month_name.moveBy(((end_frac - start_frac) / 2 + start_frac) * self.width - \
                            rect.width() / 2, start_y + 2)
            month_name.start_frac = start_frac
            month_name.end_frac = end_frac
            self.month_labels.append(month_name)
            self.addToGroup(month_name)

    def plotEvents(self):
        starttime = self.env.starttime
        endtime = self.env.endtime
        time_range = float(endtime - starttime)
        events = self.env.db.getEventsInTimeSpan()
        start_y = 45
        self.events = []
        for event in events:
            try:
                magnitude = float(event['magnitude'])
            except:
                magnitude = 1
            if magnitude < 1:
                magnitude = 1
            magnitude -= 1
            size_span = self.max_event_size - self.min_event_size
            size = magnitude / float(4) * size_span + self.min_event_size
            offset = (15 - size) / float(2)
            time = event['origin_time']
            frac = (time - starttime) / time_range
            ev = QtGui.QGraphicsEllipseItem(frac * self.width - size / 2, start_y + offset, size, size)
            ev.fraction = frac
            ev.event_size = size
            if event['event_type'] == 'manual':
                ev.setBrush(self.event_color_manual)
                ev.setPen(self.event_color_manual_dark)
                ev.setZValue(50)
            else:
                ev.setBrush(self.event_color_automatic)
                ev.setPen(self.event_color_automatic_dark)
                ev.setZValue(40)
            ev.setToolTip('Event: %s\nMagnitude: %s (%s)\n%s\nLat: %s, Lon: %s' \
                  % (event['event_id'], event['magnitude'], event['event_type'],
                     event['origin_time'], event['origin_latitude'],
                     event['origin_longitude']))
            self.addToGroup(ev)
            self.events.append(ev)

    def dayScale(self, top=True):
        """
        Creates the subdivisions of the month scale.
        """
        color = self.color1
        starttime = self.env.starttime
        endtime = self.env.endtime
        time_range = float(endtime - starttime)
        # Pixel counts.
        start_x = 0
        end_x = self.width
        x_range = end_x - start_x
        # Top or bottom year scale.
        start_y = 30
        end_y = 45
        y_range = self.subscale_height
        # Get the number of days.
        starttime_day = UTCDateTime(starttime.year, starttime.month,
                                    starttime.day)
        endtime_day = UTCDateTime(endtime.year, endtime.month,
                                    endtime.day)
        days = int((endtime_day - starttime_day) / 86400) + 1
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
            if not day % 2:
                day_box = QtGui.QGraphicsRectItem(start, start_y,
                            graph_width, y_range)
                day_box.start_frac = start_frac
                day_box.end_frac = end_frac
                day_box.setBrush(color)
                day_box.setZValue(-198)
                self.addToGroup(day_box)
                # Add to list for easier tracking.
                self.day_boxes.append(day_box)
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
            day_name.moveBy(((end_frac - start_frac) / 2 + start_frac) * self.width - \
                            rect.width() / 2, start_y + 2)
            self.addToGroup(day_name)
            day_name.start_frac = start_frac
            day_name.end_frac = end_frac
            self.day_labels.append(day_name)
