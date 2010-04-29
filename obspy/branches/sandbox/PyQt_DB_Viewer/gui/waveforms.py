from PyQt4 import QtCore, QtGui, QtOpenGL
from random import randint
from networks import TreeSelector


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
        self.x_start = self.hmargin
        self.y_start = y_start
        self.plot_height = height
        self.plot_width = width - 2*self.hmargin
        self.stream = stream
        self.scene = scene
        self.count = self.stream[0].stats.npts
        super(PreviewPlot, self).__init__()
        self.background_box = QtGui.QGraphicsRectItem(self.x_start,
                                self.y_start, self.plot_width, self.plot_height)
        # Set the title of the plot.
        self.title = title
        # Set background color.
        self.background_box.setBrush(QtGui.QColor(200,200,200,255))
        self.background_box.setZValue(-200)
        self.addToGroup(self.background_box)
        # Create button.
        self.button = QtGui.QPushButton(self.title)
        self.button.move(8, self.y_start + 3)
        self.menu = QtGui.QMenu(self.button)
        self.delete_action = self.menu.addAction('Delete Trace')
        self.button.setMenu(self.menu)
        self.button.setStyleSheet("border: 1px solid #000000;")
        self.button_in_scene = scene.addWidget(self.button)
        self.button_in_scene.setZValue(10)
        # Create a menu for the Button.
        step = self.plot_width/float(self.count)
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
            half_height = self.plot_height/2.0 - 0.5
            # Let the maximum be at half_height
            # The 0.5 is there because later we will add 0.5 to everything to
            # also show very small amplitudes.
            factor = 1/data_max * (half_height - 0.5)
            data[data>=0] *= factor
            bottom = half_height - data
            top = 2*data + 1
            for _j in xrange(self.count):
                if data[_j] < 0:
                    continue
                self.bars.append(QtGui.QGraphicsRectItem(self.x_start + _j*step,
                            y_start + bottom[_j], step, top[_j], self))
                # Fill with black color and disable the pen.
                self.bars[-1].setBrush(QtGui.QColor(0,0,0,255))
                self.bars[-1].setPen(QtGui.QPen(QtCore.Qt.PenStyle(0)))
                self.bars[-1].start_percentage =\
                    (_j*step)/float(self.plot_width)
                self.addToGroup(self.bars[-1])
        self.events = []
        # Plot events.
        picks = self.env.db.getPicksForChannel(self.title)
        if picks:
            for pick in picks:
                time = pick['time']
                # The percent location in plot width of the pick.
                percent = (time - self.env.starttime)/self.env.time_range
                x = percent * self.plot_width
                line = QtGui.QGraphicsLineItem(self.x_start + x,
                        self.y_start+1, self.x_start + x,
                        self.y_start + self.plot_height -1, self)
                self.events.append(line)
                if pick['event_type'] == 'automatic':
                    if 'p' in pick['phaseHint'].lower():
                        line.setPen(QtGui.QColor(255,0,0,200))
                    else:
                        line.setPen(QtGui.QColor(200,0,0,200))
                    line.setZValue(-100)
                else:
                    if 'p' in pick['phaseHint'].lower():
                        line.setPen(QtGui.QColor(0,255,0,200))
                    else:
                        line.setPen(QtGui.QColor(0,200,0,200))
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
        QtCore.QObject.connect(self.delete_action, QtCore.SIGNAL("triggered(bool)"), self.delete)

    def deleteItems(self):
        self.scene.removeItem(self)
        self.button.deleteLater()

    def delete(self, *args):
        self.deleteItems()
        # Call the parent to delete the reference to this particular plot. The
        # garbage collector should be able to take care of the rest.
        self.scene.removeFromWaveformList(self.title)

    def resize(self, width, height):
        """
        Resizes the single preview.
        """
        self.plot_width = width - 2*self.hmargin
        box_rect = self.background_box.rect()
        self.background_box.setRect(box_rect.x(), box_rect.y(), self.plot_width,
                     self.plot_height)
        # Resize all bars. Slow but this does not have to be fast. Using
        # transformations also works and would be a cleaner solution but
        # scrolling transformated items is painfully slow.
        step = self.plot_width/float(self.count)
        for _i, bar in enumerate(self.bars):
            rect = bar.rect()
            bar.setRect(self.x_start + bar.start_percentage * self.plot_width, rect.y(), step, rect.height())
        # Move all events.
        if hasattr(self, 'events'):
            for event in self.events:
                x = event.percent * self.plot_width
                event.setLine(self.x_start + x,
                        self.y_start+1, self.x_start + x,
                        self.y_start + self.plot_height -1)

class WaveformScene(QtGui.QGraphicsScene):
    """
    Scene that actually displays the waveforms.
    """
    def __init__(self, env, *args, **kwargs):
	# Some problem with kwargs with Python 2.5.2 and PyQt 4.4
        # super(WaveformScene, self).__init__(*args, **kwargs)
        super(WaveformScene, self).__init__()
        self.env = env
        self.position = 0
        # List that keeps track of waveform objects.
        self.waveforms = []
        self.hmargin = kwargs.get('hmargin', 5)
        self.vmargin = kwargs.get('vmargin', 5)
        self.plot_height = kwargs.get('plot_height', 50)

    def add_channel(self, new_sel, old_sel):
        """
        Is connected to selectionChanged signal. The first argument is the new
        selection and the second the old selection.
        """
        # This is the path and the type of the requested waveform.
        # e.g. ('BW', 'network') or ('BW.FURT..EHE', 'channel')
        path = new_sel.indexes()[0].model().getFullPath(new_sel.indexes()[0])
        # Do not plot already plotted channels.
        current_items = [waveform.title for waveform in self.waveforms]
        if path[0] in current_items:
            return
        # Only plot channels.
        if path[1] != 'channel':
            return
        self.addPlot(path[0])

    def addPlot(self, channel_id):
        # Get the stream item.
        network, station, location, channel = channel_id.split('.')
        stream = self.env.handler.getItem(network, station, location, channel)
        #try:
        print stream['org_stream']
        preview = PreviewPlot(self.vmargin + len(self.waveforms)
                      * (self.plot_height + self.vmargin),
                      self.plot_height, self.hmargin, self.width(),
                      channel_id, stream['minmax_stream'], self.env, self)
        self.addItem(preview)
        self.waveforms.append(preview)
        # Manually update the scene Rectangle.
        count = len(self.waveforms)
        height = (count+1)*self.vmargin + count*self.plot_height
        self.setSceneRect(0,0,self.width(), height)


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
            if _i<index:
                continue
            waveform.moveBy(0,-amount)
            waveform.button_in_scene.moveBy(0, -amount)

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
        height = (count+1)*self.vmargin + count*self.plot_height
        self.setSceneRect(0,0,width, height)

class Waveforms(QtGui.QGraphicsView):
    def __init__(self, env, parent = None, *args, **kwargs):
        super(Waveforms, self).__init__(parent)
        self.env = env
        # Force OpenGL rendering! Super smooth scrolling for 50 plots with 1000
        # bars each. Any kind of anti aliasing is useless because only
        # right angles exist.
        # XXX: Might need to check availability on platform and fall back to
        # rasterization.
        self.setViewport(QtOpenGL.QGLWidget())
        # Init scene and connect to Viewport.
        self.scene = WaveformScene(env = self.env)
        self.setScene(self.scene)

    def resizeEvent(self, event):
        """
        Gets called every time the viewport changes sizes. Make sure to call
        the ancestors resize event to handle all eventualities.
        """
        #XXX: Very ugly: Does not work in Qt 4.4!!!
        #super(Waveforms, self).resizeEvent(event) # The event sizes method also accounts for scroll bars and the like.
        size = event.size()
        # Finally resize the scene.
        self.scene.resize(size.width(), size.height())
