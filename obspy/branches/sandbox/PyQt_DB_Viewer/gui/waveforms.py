from PyQt4 import QtCore, QtGui, QtOpenGL
from random import randint
from networks import TreeSelector


class PreviewPlot(QtGui.QGraphicsRectItem):
    """
    Single preview plot item.
    """
    def __init__(self, y_start, height, hmargin, width, title, stream, count,
                 env):
        """
        Init.
        """
        self.env = env
        self.hmargin = hmargin
        self.x_start = self.hmargin
        self.y_start = y_start
        self.plot_height = height
        self.plot_width = width - 2*self.hmargin
        self.count = count
        self.stream = stream
        super(PreviewPlot, self).__init__(self.x_start, self.y_start,
                                          self.plot_width, self.plot_height)
        # Set the title of the plot.
        self.title = title
        # Set background color.
        self.setBrush(QtGui.QColor(200,200,200,255))
        # Create title and move it.
        self.channel_id = QtGui.QGraphicsSimpleTextItem(self.title, self)
        self.channel_id.moveBy(10, self.y_start + 5)
        self.channel_id.setZValue(100)
        # It is possible to set a tooltip here.
        #self.channel_id.setToolTip('asd')
        # Create a box around the title. Make it the same size as the text.
        self.id_box = QtGui.QGraphicsRectItem(self.channel_id.boundingRect(), self)
        self.id_box.setBrush(QtGui.QColor(255,255,255,200))
        self.id_box.setZValue(99)
        self.id_box.moveBy(8, self.y_start + 3)
        #Create some random data.
        step = self.plot_width/float(self.count)
        self.bars = []
        data = self.stream[0].data
        half_height = self.plot_height/2.0
        data /= data.max()/half_height
        bottom = half_height - data
        double = 2*data
        for _j in xrange(self.count):
            self.bars.append(QtGui.QGraphicsRectItem(self.x_start + _j*step,
                                                     y_start + bottom[_j], step,
                        double[_j], self))
            # Fill with black color and disable the pen.
            self.bars[-1].setBrush(QtGui.QColor(0,0,0,255))
            self.bars[-1].setPen(QtGui.QPen(QtCore.Qt.PenStyle(0)))
        self.events = []
        # Plot events.
        picks = self.env.db.getPicksForChannel(self.title)
        if picks:
            for pick in picks:
                print pick
                time = pick['time']
                # The percent location in plot width of the pick.
                percent = (time - self.env.starttime)/self.env.time_range
                x = percent * self.plot_width
                circle = QtGui.QGraphicsEllipseItem(self.x_start + x,
                                            y_start + 40, 8, 8, self)
                self.events.append(circle)
                circle.setBrush(QtGui.QColor(255,0,0,200))
                circle.setZValue(98)
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
                circle.setToolTip(tip)
                # Add percent value to circle object.
                circle.percent = percent


    def resize(self, width, height):
        """
        Resizes the single preview.
        """
        self.plot_width = width - 2*self.hmargin
        self.setRect(self.x_start, self.y_start, self.plot_width,
                     self.plot_height)
        # Resize all bars. Slow but this does not have to be fast. Using
        # transformations also works and would be a cleaner solution but
        # scrolling transformated items is painfully slow.
        count = len(self.bars)
        step = self.plot_width/float(count)
        for _i, bar in enumerate(self.bars):
            rect = bar.rect()
            bar.setRect(self.x_start + _i*step, rect.y(), step, rect.height())
        # Move all events.
        if hasattr(self, 'events'):
            for event in self.events:
                x = event.percent * self.plot_width
                event.setRect(self.x_start + x, self.y_start + 40, 8, 8)


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
	print 'Add Waveform'
        # This is the path and the type of the requested waveform.
        # e.g. ('BW', 'network') or ('BW.FURT..EHE', 'channel')
        path = new_sel.indexes()[0].model().getFullPath(new_sel.indexes()[0])
        # Only plot channels.
        if path[1] != 'channel':
            return
        # Get the stream item.
        network, station, location, channel = path[0].split('.')
        stream = self.env.handler.getItem(network, station, location, channel)
	try:
            preview = PreviewPlot(self.vmargin + len(self.waveforms)
            		      * (self.plot_height + self.vmargin),
            		      self.plot_height, self.hmargin, self.width(),
            		      path[0], stream['minmax_stream'],
            		      self.env.detail, self.env)
            self.addItem(preview)
            self.waveforms.append(preview)
            # Manually update the scene Rectangle.
            count = len(self.waveforms)
            height = (count+1)*self.vmargin + count*self.plot_height
            self.setSceneRect(0,0,self.width(), height)
	except:
	    msg = 'Error plotting/reveiving %s' % path[0]
	    self.env.st.showMessage(msg, 5000)

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
