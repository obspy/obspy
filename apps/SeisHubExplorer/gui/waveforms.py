# -*- coding: utf-8 -*-

from PyQt4 import QtCore, QtGui, QtOpenGL
from PyQt4.QtCore import Qt

from fnmatch import fnmatch
from time import time

from time_scale import TimeScale


class PreviewPlot(QtGui.QGraphicsItemGroup):
    """
    Single preview plot item.
    """
    def __init__(self, y_start, height, hmargin, width, title, stream, env,
                 scene):
        """
        Init.
        """
        self.env = env
        self.hmargin = hmargin
        self.x_start = self.hmargin - 1
        self.y_start = y_start
        self.plot_height = height
        self.plot_width = width - 2 * self.hmargin + 3
        self.stream = stream
        self.scene = scene
        self.count = self.stream[0].stats.npts
        QtGui.QGraphicsItemGroup.__init__(self)
        #super(PreviewPlot, self).__init__()
        self.background_box = QtGui.QGraphicsRectItem(self.x_start,
                                self.y_start, self.plot_width, self.plot_height)
        # Set the title of the plot.
        self.title = title
        # Set background color.
        self.background_box.setBrush(QtGui.QColor(200, 200, 200, 255))
        self.background_box.setZValue(-200)
        self.addToGroup(self.background_box)
        # Create button.
        self.button = QtGui.QPushButton('X')
        self.button.move(3, self.y_start + 3)
        #self.menu = QtGui.QMenu(self.button)
        #self.delete_action = self.menu.addAction('Delete Trace')
        #self.button.setMenu(self.menu)
        self.button.setStyleSheet("border: 1px solid #000000;")
        self.button_in_scene = scene.addWidget(self.button)
        self.button_in_scene.setZValue(10)
        step = self.plot_width / float(self.count)

        self.title_label = QtGui.QLabel(self.title)
        self.title_label.move(15, self.y_start + 3)
        self.title_in_scene = scene.addWidget(self.title_label)
        self.title_in_scene.setZValue(10)

        # Plot the preview data.
        self.bars = []
        data = self.stream[0].data
        data_max = data.max()
        self.connectSlots()
        if self.env.debug:
            print '==================================='
            print 'Actually plotted trace:'
            print self.stream
            print '==================================='
        if data_max != -1:
            # Middle of the graphic.
            half_height = self.plot_height / 2.0 - 0.5
            # Let the maximum be at half_height
            # The 0.5 is there because later we will add 0.5 to everything to
            # also show very small amplitudes.
            factor = 1 / data_max * (half_height - 0.5)
            data[data >= 0] *= factor
            bottom = half_height - data
            top = 2 * data + 1
            for _j in xrange(self.count):
                if data[_j] < 0:
                    continue
                self.bars.append(QtGui.QGraphicsRectItem(self.x_start + _j * step,
                            y_start + bottom[_j], step, top[_j], self))
                # Fill with black color and disable the pen.
                self.bars[-1].setBrush(QtGui.QColor(0, 0, 0, 255))
                self.bars[-1].setPen(QtGui.QPen(QtCore.Qt.PenStyle(0)))
                self.bars[-1].start_percentage = \
                    (_j * step) / float(self.plot_width)
                self.addToGroup(self.bars[-1])
        self.events = []
        # Plot events.
        picks = self.env.db.getPicksForChannel(self.title)
        if picks:
            for pick in picks:
                time = pick['time']
                # The percent location in plot width of the pick.
                percent = (time - self.env.starttime) / self.env.time_range
                x = percent * self.plot_width
                line = QtGui.QGraphicsLineItem(self.x_start + x,
                        self.y_start + 1, self.x_start + x,
                        self.y_start + self.plot_height - 1, self)
                self.events.append(line)
                if pick['event_type'] == 'automatic':
                    if 'p' in pick['phaseHint'].lower():
                        line.setPen(QtGui.QColor(255, 0, 0, 200))
                    else:
                        line.setPen(QtGui.QColor(200, 0, 0, 200))
                    line.setZValue(-100)
                else:
                    if 'p' in pick['phaseHint'].lower():
                        line.setPen(QtGui.QColor(0, 255, 0, 200))
                    else:
                        line.setPen(QtGui.QColor(0, 200, 0, 200))
                    line.setZValue(-90)
                # Set tool tip.
                tip = '<b>%s</b><br>%s<br>Phase: %s, Polarity: %s' % \
                        (pick['event_id'], pick['time'],
                         pick['phaseHint'],
                         pick['polarity'])
                tip += '<br><br><b>Event</b><br>Coods: %.3f, %.3f, %.0f<br>' % \
                        (pick['origin_latitude'], pick['origin_longitude'],
                         pick['origin_depth'])
                tip += 'Magnitude: %.2f (%s)<br>Time: %s' % \
                        (pick['magnitude'], pick['magnitude_type'],
                         pick['origin_time'])
                line.setToolTip(tip)
                # Add percent value to circle object.
                line.percent = percent
                self.addToGroup(line)

    def connectSlots(self):
        """
        Connect the slots of the PreviewPlot.
        """
        QtCore.QObject.connect(self.button, QtCore.SIGNAL("pressed()"), self.delete)
        pass

    def deleteItems(self):
        self.scene.removeItem(self)
        self.button.deleteLater()
        self.title_label.deleteLater()

    def delete(self, *args):
        self.deleteItems()
        # Call the parent to delete the reference to this particular plot. The
        # garbage collector should be able to take care of the rest.
        self.scene.removeFromWaveformList(self.title)

    def resize(self, width, height):
        """
        Resizes the single preview.
        """
        self.plot_width = width - 2 * self.hmargin + 3
        box_rect = self.background_box.rect()
        self.background_box.setRect(box_rect.x(), box_rect.y(), self.plot_width,
                     self.plot_height)
        # Resize all bars. Slow but this does not have to be fast. Using
        # transformations also works and would be a cleaner solution but
        # scrolling transformated items is painfully slow.
        step = self.plot_width / float(self.count)
        for _i, bar in enumerate(self.bars):
            rect = bar.rect()
            bar.setRect(self.x_start + bar.start_percentage * self.plot_width, rect.y(), step, rect.height())
        # Move all events.
        if hasattr(self, 'events'):
            for event in self.events:
                x = event.percent * self.plot_width
                event.setLine(self.x_start + x,
                        self.y_start + 1, self.x_start + x,
                        self.y_start + self.plot_height - 1)


class WaveformScene(QtGui.QGraphicsScene):
    """
    Scene that actually displays the waveforms.
    """
    def __init__(self, env, *args, **kwargs):
	# Some problem with kwargs with Python 2.5.2 and PyQt 4.4
        # super(WaveformScene, self).__init__(*args, **kwargs)
        #super(WaveformScene, self).__init__()
        QtGui.QGraphicsScene.__init__(self)
        self.env = env
        self.position = 0
        # List that keeps track of waveform objects.
        self.waveforms = []
        self.hmargin = kwargs.get('hmargin', 0)
        self.vmargin = kwargs.get('vmargin', 3)
        self.plot_height = kwargs.get('plot_height', 50)
        # Variables to keep track of the mouse.
        self.last_clicked_time = time()
        self.start_mouse_pos = None
        self.selection_box = None
        self.mouse_moved = False

    # Override the mouse events to handle the selection of a time-span.
    def mousePressEvent(self, event):
        """
        Save the mouse position.
        """
        self.start_mouse_pos = event.scenePos()
        QtGui.QGraphicsScene.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        """
        Draw the selection rectangle if the mouse if moved. By default this
        event is only fired if the mouse button is held down.
        """
        if event.buttons() == Qt.NoButton:
            return
        self.mouse_moved = True
        # Delete the old selection box.
        if self.selection_box:
            self.removeItem(self.selection_box)
        # Draw the new one.
        # XXX: Maybe just move/resize the old one??
        x = self.start_mouse_pos.x()
        y = -1000
        width = event.scenePos().x() - x
        height = 2000
        # Take care of negative width and heights.
        rect = QtCore.QRectF(x, y, width, height)
        rect = rect.normalized()
        self.selection_box = QtGui.QGraphicsRectItem(rect)
        self.selection_box.setZValue(1000000)
        self.selection_box.setBrush(QtGui.QColor(0, 0, 0, 20))
        # XXX: This sometimes is not immediatly updates. Find out why!
        self.addItem(self.selection_box)
        self.selection_box.show()
        # Figure out the corresponding times.
        scene_width = self.sceneRect().width()
        starttime = self.env.starttime + rect.x() / float(scene_width) * self.env.time_range
        endtime = self.env.starttime + (rect.x() + rect.width()) / float(scene_width) * self.env.time_range
        self.selection_box.starttime = starttime
        self.selection_box.endtime = endtime
        # Set the message.
        self.env.st.showMessage('%s - %s [%.1f minutes]' %
                                (starttime.strftime('%Y-%m-%dT%H:%M'),
                                 endtime.strftime('%Y-%m-%dT%H:%M'),
                                 (endtime - starttime) / 60.0))
        self.env.main_window.waveforms.refreshOpenGL()

    def mouseReleaseEvent(self, event):
        """
        Fired every time the mouse is released.
        """
        if self.mouse_moved:
            msg = 'Left click in selected area to zoom in, ' + \
                  'right click to send to picker.'
            self.env.main_window.plotStatus.setText(msg)
            self.mouse_moved = False
            return
        else:
            # Figure out whether or not the button is clicked in the box.
            if self.selection_box and \
               self.start_mouse_pos.x() > self.selection_box.boundingRect().x()\
               and\
               self.start_mouse_pos.x() < (self.selection_box.boundingRect().x() + \
               self.selection_box.boundingRect().width()):
                # If the first mouse button is pressed, zoom in.
                if event.button() == Qt.LeftButton:
                    self.env.main_window.changeTimes(self.selection_box.starttime,
                                                     self.selection_box.endtime)
                elif event.button() == Qt.RightButton:
                    self.env.main_window.callPicker(self.selection_box.starttime,
                                                    self.selection_box.endtime)
            elif time() - self.last_clicked_time < self.env.double_click_time:
                self.maxZoom(event)
            self.last_clicked_time = time()
            self.removeItem(self.selection_box)
            self.removePlotStatus()
            self.env.st.showMessage('')

    def maxZoom(self, mouse_event):
        """
        Sets the zoom to the maximum zoom level centered around the mouse
        event.
        """
        scene_width = self.sceneRect().width()
        x_pos = mouse_event.scenePos().x()
        # Determine the time at the click.
        middle_time = self.env.starttime + x_pos / float(scene_width) * self.env.time_range
        # Get the start-and endtimes.
        starttime = middle_time - 0.5 * self.env.maximum_zoom_level
        endtime = middle_time + 0.5 * self.env.maximum_zoom_level
        self.env.main_window.changeTimes(starttime, endtime)

    def startup(self):
        """
        The Timescale can only be drawn once the scene is being used by a view.
        Otherwise the width of the scene is 0
        """
        pass

    def add_channel(self, new_sel, old_sel):
        """
        Is connected to selectionChanged signal. The first argument is the new
        selection and the second the old selection.
        """
        # This is the path and the type of the requested waveform.
        # e.g. ('BW', 'network') or ('BW.FURT..EHE', 'channel')
        # XXX: Win32 crashes here
        print new_sel.indexes()
        path = new_sel.indexes()[0].model().getFullPath(new_sel.indexes()[0])
        # Only plot channels.
        if path[1] != 'channel' and path[1] != 'channel_list':
            return
        if path[1] == 'channel_list':
            self.add_channel_list(path[0])
            return
        self.addPlot(path[0])

    def add_channel_list(self, channel_name):
        """
        Plots a channel list.
        """
        channels = self.env.channel_lists[channel_name]
        for channel in channels:
            expanded_list = []
            # Check for wildcards and create channels.
            for chan in self.env.all_channels:
                if fnmatch(chan, channel):
                    expanded_list.append(chan)
            expanded_list.sort()
            # Plot them.
            for chan in expanded_list:
                self.addPlot(chan)

    def addPlot(self, channel_id):
        # Do not plot already plotted channels.
        current_items = [waveform.title for waveform in self.waveforms]
        if channel_id in current_items:
            return
        # Get the stream item.
        network, station, location, channel = channel_id.split('.')
        stream = self.env.handler.getItem(network, station, location, channel)
        #try:
        preview = PreviewPlot(self.vmargin + len(self.waveforms)
                      * (self.plot_height + self.vmargin),
                      self.plot_height, self.hmargin, self.width(),
                      channel_id, stream['minmax_stream'], self.env, self)
        self.addItem(preview)
        self.waveforms.append(preview)
        # Manually update the scene Rectangle.
        count = len(self.waveforms)
        height = (count + 1) * self.vmargin + count * self.plot_height
        self.setSceneRect(0, 0, self.width(), height)

    def removeFromWaveformList(self, title):
        """
        Removes the waveform with title title from self.waveforms.
        """
        titles = [waveform.title for waveform in self.waveforms]
        index = titles.index(title)
        del self.waveforms[index]
        # Shift all further traces.
        amount = (self.plot_height + self.vmargin)
        for _i, waveform in enumerate(self.waveforms):
            if _i < index:
                continue
            waveform.moveBy(0, -amount)
            waveform.button_in_scene.moveBy(0, -amount)
            waveform.title_in_scene.moveBy(0, -amount)

    def redraw(self):
        """
        Redraws everything with updated times.
        """
        items = [waveform.title for waveform in self.waveforms]
        for waveform in self.waveforms:
            waveform.deleteItems()
        self.waveforms = []
        for item in items:
            self.addPlot(item)

    def resize(self, width, height):
        """
        Handles resizing.
        """
        for waveform in self.waveforms:
            waveform.resize(width, height)
        # Manually update the scene Rectangle.
        count = len(self.waveforms)
        height = (count + 1) * self.vmargin + count * self.plot_height
        self.setSceneRect(0, 0, width, height)

    def setPlotStatus(self, msg):
        """
        Sets the plot status to msg.
        """
        self.env.main_window.plotStatus.setText(msg)

    def removePlotStatus(self):
        """
        Sets the plot status to its default value.
        """
        msg = 'Drag to select a time frame. Doubleclick anywhere to zoom to ' + \
              'maximal zoom level.'
        self.env.main_window.plotStatus.setText(msg)


class TimescaleScene(QtGui.QGraphicsScene):
    def __init__(self, env):
        QtGui.QGraphicsScene.__init__(self)
        #super(TimescaleScene, self).__init__()
        self.env = env

    def startup(self):
        # Add the time scale.
        self.time_scale = TimeScale(self.env, self, self.view.width())
        self.addItem(self.time_scale)

    def redraw(self):
        # Delete time scale.
        self.removeItem(self.time_scale)
        # Create new time scale.
        self.time_scale = TimeScale(self.env, self, self.width())
        self.addItem(self.time_scale)

    def resize(self, width, height):
        # Check if available.
        if hasattr(self, 'time_scale'):
            self.time_scale.resize(self.view.width(), height)
        # Manually update the scene Rectangle.
        self.setSceneRect(0, 0, width, 60)


class TimescaleView(QtGui.QGraphicsView):
    """
    View for the timescale.
    """
    def __init__(self, env):
        QtGui.QGraphicsView.__init__(self)
        #super(TimescaleView, self).__init__()
        self.env = env
        # Force OpenGL rendering! Super smooth scrolling for 50 plots with 1000
        # bars each. Any kind of anti aliasing is useless because only
        # right angles exist.
        if not self.env.software_rendering:
            self.opengl_widget = QtOpenGL.QGLWidget()
            self.setViewport(self.opengl_widget)
        # This is manually adjusted.
        self.setMinimumHeight(62)
        self.setMaximumHeight(64)

    def resizeEvent(self, event):
        """
        Gets called every time the viewport changes sizes. Make sure to call
        the ancestors resize event to handle all eventualities.
        """
        size = event.size()
        # Finally resize the scene.
        self.scene.resize(size.width(), size.height())


class WaveformView(QtGui.QGraphicsView):
    """
    View for the waveform.
    """
    def __init__(self, env):
        QtGui.QGraphicsView.__init__(self)
        #super(WaveformView, self).__init__()
        self.env = env
        # Force OpenGL rendering! Super smooth scrolling for 50 plots with 1000
        # bars each. Any kind of anti aliasing is useless because only
        # right angles exist.
        if not self.env.software_rendering:
            self.opengl_widget = QtOpenGL.QGLWidget()
            self.setViewport(self.opengl_widget)

    def resizeEvent(self, event):
        """
        Gets called every time the viewport changes sizes. Make sure to call
        the ancestors resize event to handle all eventualities.
        """
        size = event.size()
        # Finally resize the scene.
        self.scene.resize(size.width(), size.height())


class Waveforms(QtGui.QFrame):
    """
    A Frame that contains two view that will show the time scale and the
    waveforms.
    """
    def __init__(self, env, parent=None, *args, **kwargs):
        QtGui.QFrame.__init__(self, parent)
        #super(Waveforms, self).__init__(parent)
        self.env = env
        self._setupInterface()
        # Call once to always set the default value.
        self.waveform_scene.removePlotStatus()

    def _setupInterface(self):
        """
        Build everything inside the frame.
        """
        # Init the layout.
        self.layout = QtGui.QVBoxLayout()
        self.layout.setMargin(0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)
        # Create a timescale and a waveform view.
        self.timescale_view = TimescaleView(self.env)
        self.waveform_view = WaveformView(self.env)
        # Add the view to the layout.
        self.layout.addWidget(self.timescale_view)
        self.layout.addWidget(self.waveform_view)
        # Setup both scenes.
        self.timescale_scene = TimescaleScene(self.env)
        self.waveform_scene = WaveformScene(self.env)
        # Give both the scene and the view access to each other.
        self.timescale_view.scene = self.timescale_scene
        self.waveform_view.scene = self.waveform_scene
        self.timescale_scene.view = self.timescale_view
        self.waveform_scene.view = self.waveform_view
        # Connect scenes and views.
        self.timescale_view.setScene(self.timescale_scene)
        self.waveform_view.setScene(self.waveform_scene)

    def refreshOpenGL(self):
        """
        Redraws the OpenGL scenes for more fluid updating. Will only be
        executed of software rendering is off.
        """
        if not self.env.software_rendering:
            self.timescale_view.opengl_widget.paintGL()
            self.waveform_view.opengl_widget.paintGL()

    def redraw(self):
        self.timescale_scene.redraw()
        self.waveform_scene.redraw()

    def startup(self):
        self.timescale_scene.startup()
