class SceneGroup(pyglet.graphics.Group):
    """
    Scene group
    """

    def __init__(self, parent=None):
        """
        Create scene group.
        """
        pyglet.graphics.Group.__init__(self, parent)
        
    def set_state(self):
        """
        Setup pojection matrix and polygon offset fill mode.
        """
        # XXX: Figure out what the next four calls exactly do and if they are
        # necessary/could be replaced with cheaper calls.
        glClearDepth(1.0)                   # Enables Clearing Of The Depth Buffer
        glDepthFunc(GL_LESS)                # The Type Of Depth Test To Do
        glEnable(GL_DEPTH_TEST)             # Enables Depth Testing
        glShadeModel(GL_SMOOTH)             # Enables Smooth Color Shading

    
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Enable backface culling.
        glEnable(GL_CULL_FACE)


        ###########################
        # SETUP THE PROJETION MATRIX
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        # Make an orthographic projetion and set a virtual resolution coordinate
        # system going from -100 to 100 in each direction.
        # The last two arguments are the near and far clipping planes.
        # XXX: Deprecated: Replace with glMultMatrix.
        glOrtho(-100.0,100.0,-100.0,100.0,0,128);
        ###########################

        glMatrixMode(gl_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

    def unset_state(self):
        """
        Unset projection matrix
        """
        glDisable(gl_DEPTH_TEST)
        glMatrixMode(gl_PROJECTION)
        glPopMatrix()
        glMatrixMode(gl_MODELVIEW)
        glPopMatrix()

    def _isLeapYear(self, year):
        """
        Returns true, when year is a leap year and false otherwise.
        """
        dt = UTCDateTime(year, 2,28,23,59)
        dt = dt + 1000
        if dt.day == 29:
            return True
        else:
            return False

    def _writeBackgroundToVertexBufferObject(self, x_start, x_end, y_start,
                                             y_end, year):
        """
        Creates a vertex buffer object and writes a background consisting of
        GL_QUADS and GL_LINES to it.

        The background will start at x_start and end at x_end. And strecht from
        y_start to y_end.

        Returns the vertex buffer object and the index where the lines start
        and how many objects are in it.
        """
        leap = self._isLeapYear(year)
        months = [['Jan', 31], ['Feb', 28], ['Mar', 31], ['Apr', 30],
                  ['May', 31], ['Jun', 30], ['Jul', 31], ['Aug', 31],
                  ['Sep', 30], ['Oct', 31], ['Nov', 30], ['Dez', 31],]
        # Account for leap year.
        if leap:
            months[1][1] += 1
        # Store quads here.
        quads = []
        # Loop over each month. The resulting quads will go from and from 0 to
        # 310 in the x direction and from 0 to 120 in the y direction.
        for _i in xrange(12):
            id = 11 - _i
            if _i%2 == 0:
                even = True
            else:
                even = False
            count = 0
            width = 10
            for _j in xrange(months[_i][1]):
                # Draw a kind of checkerboard.
                if (even and _j%2 != 0) or (not even and _j%2 == 0):
                    continue
                quads.extend([[_j * width, id*width], [(_j+1) * width, id * width],
                             [(_j+1) * width, (id+1) * width],
                             [_j * width, (id+1)*width]])
        # Store in array.
        quads = np.array(quads, dtype = 'float32')
        # Squash everything to fit in between 0 and 1.
        quads[:,0] -= quads[:,0].min()
        quads[:,0] /= quads[:,0].max()
        quads[:,1] -= quads[:,1].min()
        quads[:,1] /= quads[:,1].max()
        # Now strech to fit from -100 to 100 in every direction.
        quads[:,0] *= x_end - x_start
        quads[:,0] += x_start
        quads[:,1] *= y_end - y_start
        quads[:,1] += y_start
        # Create third coordinate.
        temp = np.zeros((len(quads), 3), dtype = 'float32')
        temp[:, 0] = quads[:, 0]
        temp[:, 1] = quads[:, 1]

        # Read into Buffer helper.
        self.length = len(quads)
        self.vertex_buffer = vbo.VBO(temp)

if __name__ == '__main__':

    width = 1200
    height = 600
    # Read and prepare the index file.
    st = read('BW.RJOB..EHE.2009.index')
    # Use log scale to better view the result.
    # XXX: This needs some improvement...how to really calculate a log scale?
    st[0].data = np.log10(st[0].data + 1)

    # Store everything in range from 0 to 1.
    st[0].data /= st[0].data.max()

    quad_length = 0

    background = glydget.Rectangle(0, height, width, height,
                                   colors= [  0,  0,127,255,   0,  0,127,255,
                                            255,255,255,255, 255,255,255,255],
                                   filled=True)


    #Get quads and store in new vertex buffer object.
    def getGraph(x_start, x_end, y_start, y_end, stream, year):
        global quad_length, data_quads
        months = [31,28,31,30,31,30,31,31,30,31,30,31]
        quads = []
        y_step = (y_end - y_start)/12.0
        day_length = (x_end - x_start)/31.0
        # Loop over each month.
        for _i in xrange(1, 13, 1):
            current_low = y_end - _i * y_step
            current_height = y_end - (_i -1) * y_step
            #getPicture(stream,UTCDateTime(year, _i, 1),
            #           UTCDateTime(year, _i + 1, 1), 1000, 80, 'blub.png')
            if _i == 12:
                st = stream.slice(UTCDateTime(year, _i, 1),
                              UTCDateTime(year + 1, 1, 1))
                # Reduce the amount of quads to draw to one tenth.
                data = st[0].data
            else:
                st = stream.slice(UTCDateTime(year, _i, 1),
                              UTCDateTime(year, _i + 1, 1))
                data = st[0].data[:-1]
            data = data.reshape(len(data)/10, 10)
            quad_length += len(data) * 12
            data = data.max(axis=1)
            data = data * (current_height - current_low) / 2
            cur_quads = np.zeros(4 * len(data) * 3, dtype='float32')
            start = x_start
            end = x_start + months[_i - 1] * day_length
            cur_quads[1::12] = data + current_low + (current_height -\
                                                     current_low)/2
            cur_quads[4::12] = data + current_low + (current_height -\
                                                     current_low)/2
            cur_quads[7::12] = -data + current_low + (current_height -\
                                                      current_low)/2
            cur_quads[10::12] = -data + current_low + (current_height -\
                                                       current_low)/2

            steps = np.linspace(start, end, len(data) + 1)

            cur_quads[0::12] = steps[1:]
            cur_quads[3::12] = steps[: -1]
            cur_quads[6::12] = steps[: -1]
            cur_quads[9::12] = steps[1:]
            
            cur_quads = cur_quads.reshape(len(data) * 4, 3)

            quads.append(cur_quads)
        quads = np.concatenate(quads)
        return vbo.VBO(quads)

    zoomed_quad_length = None

    def getZoomedGraph(x_start, x_end, y_start, y_end):
        """
        Calculate the zoomed trace quads and store them in the returned vertex
        buffer object.
        """
        global quad_length, data_quads, zoom_start, zoom_end, st,\
               zoomed_quad_length
        data = st.slice(zoom_start, zoom_end + 84599)[0].data
        quads = np.empty(len(data) * 4 * 3, dtype = 'float32')

        steps = np.linspace(x_start, x_end, len(data) + 1)

        y_axis = (y_end - y_start)/2.0
        data += data.min()
        data /= data.max()

        data *= y_axis
        y_axis += y_start

        # Start to draw the quads. The first one is the top right vertex and
        # then it goes counterclockwise around.
        # Set the x values.
        quads[0::12] = steps[:-1]
        quads[3::12] = steps[1:]
        quads[6::12] = steps[1:]
        quads[9::12] = steps[:-1]
        # Set the y values.
        quads[1::12] = y_axis + data
        quads[4::12] = y_axis + data
        quads[7::12] = y_axis - data
        quads[10::12] = y_axis - data
        # Set the z values.
        quads[2::12] = -1.0
        quads[5::12] = -1.0
        quads[8::12] = -1.0
        quads[11::12] = -1.0

        # The last step is to reshape the array.
        zoomed_quad_length = len(quads)
        quads.resize(len(quads)/3, 3)
        return vbo.VBO(quads)

    data_quads = getGraph(100,900,100,500,st,2009)


    pyglet.clock.schedule(lambda dt: None)
    # Limit framerate.
    # XXX: Deprecated. Use clock.schedule_interval instead.
    pyglet.clock.set_fps_limit(50)
    fps_display = pyglet.clock.ClockDisplay()

    # Enable blending/transparency.
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    scene = SceneGroup()
    scene._writeBackgroundToVertexBufferObject(100,900,100,500,2009)

    zoom = False
    zoom_box = False
    zoomed_trace = None

    def zoom_trace(*args, **kwargs):
        """
        Handles the zooming of any trace.
        """
        global zoom, zoom_box, zoomed_trace
        if not zoom:
            zoom = True
            zoom_box = pyglet.graphics.Batch()
            zoom_box.add( 4, pyglet.gl.GL_QUADS, None,
                       ('v3f', (10, 10, -1, 10, 510, -1,
                                910, 510, -1, 910, 10, -1)),
                       ('c4f', (0.2, 0.2, 0.2, 0.9,
                                0.2, 0.2, 0.2, 0.9,
                                0.3, 0.3, 0.3, 0.9,
                                0.0, 0.0, 0.0, 0.9))
                                        )
            zoom_box.add( 4, pyglet.gl.GL_QUADS, None,
                       ('v3f', (20, 20, -1, 20, 500, -1,
                                900, 500, -1, 900, 20, -1)),
                       ('c4f', (1.0, 1.0, 1.0, 0.9,
                                0.95, 0.95, 0.95, 0.9,
                                1.0, 1.0, 1.0, 0.9,
                                0.9, 0.9, 0.9, 0.9))
                                        )
            zoom_box.add(4, pyglet.gl.GL_LINE_LOOP, None,
                      ('v3f', (10, 10, -1, 910, 10, -1,
                               910, 510, -1, 10, 510, -1)),
                      ('c3B', (160,160,160,160,160,160,255,255,255,255,255,255)))
            zoomed_trace = getZoomedGraph(20,900,20,500)
        else:
            zoom = False
        

    def quit(*args):
        pyglet.app.exit()

    def append_child(button):
        win.child.append(glydget.Label('New label'))

    def remove_child(button):
        win.child.remove(win.child._children[-1])

    win = glydget.Window ("Choose Channels",
                          [glydget.Label('Current selection:'),
                           glydget.Label('    nothing selected'),
                           glydget.Button('Zoom', zoom_trace),
                           glydget.Label(' '),
                           glydget.Label('Dummy Tests below'),
                           glydget.Button('Remove child', remove_child),
                           glydget.Button('Add a child', append_child),
                           glydget.ToggleButton('Toggle me'),
                           glydget.Folder('Folder',
                                          glydget.VBox([glydget.HBox([glydget.Label('Value '),
                                          glydget.ScalarEntry(1)], True),
                                          glydget.HBox([glydget.Label('Value '),
                                          glydget.ScalarEntry(10)], True),
                                          glydget.HBox([glydget.Label('Value '),
                                          glydget.ScalarEntry(100)], True)])),
                           glydget.HBox([glydget.Label('Value'),
                                         glydget.Entry('Edit me...'),],True),])
    win.show()
    win.move(width - 270, window.height-9)
    window.push_handlers(win)

    

    # String for the months labels. {#0010} is a unicode paragraph break
    months_text_string = '{margin_bottom "18px"}JAN{#0010}FEB{#0010}MAR' +\
            '{#0010}APR{#0010}MAY{#0010}JUN{#0010}JUL{#0010}AUG{#0010}' +\
            'SEP{#0010}OCT{#0010}NOV{#0010}DEZ'
    months_document = pyglet.text.decode_attributed(months_text_string)
    months_layout = pyglet.text.DocumentLabel(document = months_document,
                           x = 50, y = 480, width = 0, height = 0,
                           multiline = True)


    position = ''
    position_document = pyglet.text.decode_text(position)
    position_document.set_style(0, 2, dict(font_name='Arial',
                                bold=True, font_size=40, color=(115,115,115,255)))
    position_layout = pyglet.text.DocumentLabel(document = position_document,
                      x = 760, y = 13)

    title = 'BW.RJOB..EHE'
    title_doc = pyglet.text.decode_text(title)
    title_doc.set_style(0,2, dict(font_name = 'Arial',
                                bold=True, font_size=40, color=(115,115,115,255)))
    title_layout = pyglet.text.DocumentLabel(document = title_doc,
                   x = 550, y = 580, anchor_x = 'center', anchor_y = 'top')
    
    # Create the labels for the days. Write each into the same batch to
    # accelerate rendering in the main loop.
    batch = pyglet.graphics.Batch()
    for _i in range(1,32, 1):
        start = 100
        increment = 800/31.0
        start = start + increment/2.0
        string = '%s' % _i
        doc = pyglet.text.decode_attributed(string)
        layout = pyglet.text.DocumentLabel(document = doc,
                 x = start + increment*(_i - 1),
                 y = 85, width = 1000, height = 0,
                 anchor_x = 'center', batch = batch)

    show = False
    drag_active = False
    drag_start = None
    zoom_start = None
    zoom_end = None

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        """
        Fired when mouse button released. Currently used only to reset the the
        drag-start.
        """
        global drag_start
        drag_start = None


    @window.event
    def on_mouse_drag(x, y, dx, dy, button, modifiers):
        """
        Event fired on mouse drag.
        """
        global drag_active, drag_start, zoom_start, zoom_end
        # Invert lengths due to coordinate system.
        months = [31,30,31,30,31,31,30,31,30,31,28,31]
        x_start = 100.0
        x_end = 900.0
        y_start = 100.0
        y_end = 500.0
        x_delta = (x_end - x_start) / 31
        y_delta = (y_end - y_start) / 12

        drag_active = False
        
        if x > 100.0 and x < 900.0 and y > 100.0 and y < 500.0:
            if not drag_start:
                drag_start = x
            # Get the current month.
            month = int((y - y_start) / y_delta) + 1 
            length_of_month = months[month - 1]
            # Check if in boundary for current month.
            if (x - x_start) < x_delta * length_of_month:
                drag_active = True
                createDragBox(drag_start, x ,
                                 y_start + y_delta * (month - 1), 
                                 y_start + y_delta * month)
                day1 = int((x - 100.0)/ 800.0 * 31.0) + 1
                day2 = int((drag_start - 100.0)/ 800.0 * 31.0) + 1
                if day1>day2:
                    day1, day2 = day2, day1
                if day2 > length_of_month or x < x_start or y > y_end\
                   or y < y_start:
                    win.child.children[1].text = '    nothing selected'
                else:
                    zoom_start = UTCDateTime(2009, 13 - month, day1)
                    zoom_end = UTCDateTime(2009, 13 - month, day2)
                    win.child.children[1].text = '    ' +\
                        zoom_start.strftime('%x') + \
                        '-' + \
                        zoom_end.strftime('%x')
                    

            else:
                win.child.children[1].text = '    nothing selected'


    @window.event
    def on_mouse_motion(x, y, dx, dy):
        """
        This event gets fired every time the mouse moves.
        """
        global show, position
        # Invert lengths due to coordinate system.
        months = [31,30,31,30,31,31,30,31,30,31,28,31]
        x_start = 100.0
        x_end = 900.0
        y_start = 100.0
        y_end = 500.0
        x_delta = (x_end - x_start) / 31
        y_delta = (y_end - y_start) / 12

        show = False
        
        if x > 100.0 and x < 900.0 and y > 100.0 and y < 500.0:
            # Get the current month.
            month = int((y - y_start) / y_delta) + 1 
            length_of_month = months[month - 1]
            # Check if in boundary for current month.
            if (x - x_start) < x_delta * length_of_month:
                show = True
                createColoredBox(x_start, x_start + length_of_month * x_delta ,
                                 y_start + y_delta * (month - 1), 
                                 y_start + y_delta * month)
            day = int((x - 100.0)/ 800.0 * 31.0) + 1
            if day > length_of_month:
                position_document.text = ''
            else:
                position_document.text = UTCDateTime(2009, 13 - month,
                                                     day).strftime('%x')
        else:
            position_document.text = ''
            
    def createDragBox(x_start, x_end, y_start, y_end, z = 0):
        """
        XXX: Highly inefficient method to do this. Need to find better way.
        """
        global drag_box
        drag_box = pyglet.graphics.Batch()
        drag_box.add( 4, pyglet.gl.GL_QUADS, None,
                   ('v3f', (x_start, y_start, z, x_start, y_end, z,
                            x_end, y_end, z, x_end, y_start, z)),
                   ('c4f', (0.3, 0.0, 0.0, 0.7,
                            0.3, 0.5, 0.5, 0.7,
                            0.3, 0.5, 0.5, 0.7,
                            0.3, 0.0, 0.0, 0.7))
                                    )
        drag_box.add(4, pyglet.gl.GL_LINE_LOOP, None,
                  ('v3f', (x_start, y_start, z, x_end, y_start, z,
                           x_end, y_end, z, x_start, y_end, z)),
                  ('c3B', (0,0,0,0,0,0,0,0,0,0,0,0)))

    def createColoredBox(x_start, x_end, y_start, y_end, z = 0):
        """
        XXX: Highly inefficient method to do this. Need to find better way.
        """
        global shade
        shade = pyglet.graphics.Batch()
        shade.add( 4, pyglet.gl.GL_QUADS, None,
                   ('v3f', (x_start, y_start, z, x_start, y_end, z,
                            x_end, y_end, z, x_end, y_start, z)),
                   ('c4f', (1.0, 0.0, 0.0, 0.2,
                            1.0, 0.5, 0.5, 0.2,
                            1.0, 0.5, 0.5, 0.2,
                            1.0, 0.0, 0.0, 0.5))
                                    )
        shade.add(4, pyglet.gl.GL_LINE_LOOP, None,
                  ('v3f', (x_start, y_start, z, x_end, y_start, z,
                           x_end, y_end, z, x_start, y_end, z)),
                  ('c3B', (0,0,0,0,0,0,0,0,0,0,0,0)))
        
    @window.event
    def on_draw(*args, **kwargs):
        """
        The actual main loop. Do no intesive calculations in there.
        """
        global show, data_quads, drag_active, zoom, zoom_box, zoomed_trace,\
               zoomed_quad_length
        glClearColor(0.4,0.4,0.4,0.4)
        window.clear()
        background.batch.draw()
        # Use the vertex buffer.
        glColor4f(1.0, 1.0, 1.0, 1.0)
        scene.vertex_buffer.bind() 
        glEnableClientState(GL_VERTEX_ARRAY);
        ogl.glVertexPointerf(scene.vertex_buffer)
        glDrawArrays(GL_QUADS, 0, 2 * scene.length)
        scene.vertex_buffer.unbind()

        glColor4f(0.8, 0.2, 0.2, 1.0)
        data_quads.bind()
        glEnableClientState(GL_VERTEX_ARRAY);
        ogl.glVertexPointerf(data_quads)
        glDrawArrays(GL_QUADS, 0, quad_length - 12)
        data_quads.unbind()

        if show:
            shade.draw()

        if drag_active:
            drag_box.draw()

        if zoom:
            zoom_box.draw()
            glColor4f(0.8, 0.2, 0.2, 1.0)
            zoomed_trace.bind()
            glEnableClientState(GL_VERTEX_ARRAY);
            ogl.glVertexPointerf(zoomed_trace)
            glDrawArrays(GL_QUADS, 0, zoomed_quad_length)
            zoomed_trace.unbind()
        else:
            months_layout.draw()
            batch.draw()
            position_layout.draw()

        win.batch.draw()
        #fps_display.draw()
        title_layout.draw()
        pyglet.graphics.draw(2, pyglet.gl.GL_POINTS,
                                 ('v2i', (10, 15, 30, 35))
                            )

