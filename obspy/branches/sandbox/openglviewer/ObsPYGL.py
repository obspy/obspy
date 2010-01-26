#!/usr/bin/env python

from obspy.core import read, Stream, Trace
# Allows for faster calls!
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import sys
from time import time

#The OpenGL.arrays.vbo.VBO class is a convenience wrapper 
#which makes it easier to use Vertex Buffer Objects from within 
#PyOpenGL.  It takes care of determining which implementation 
#to use, the creation of offset objects, and even basic slice-based 
#updating of the content in the VBO.
from OpenGL.arrays import vbo

class OpenGLViewer(object):
    """
    Allows viewing of Stream objects with Open GL.
    """

    def __init__(self, *args, **kwargs):
        """
        Set some class variables and Open GL calls.
        """
        # If it is a Stream object, convert to trace.
        # XXX: TODO
        # Set all variables necessary for movement.
        self._initializeMovementVariables()
        # Sets verious other class variables.
        self._setMiscVariables()
        # Map the keys to ASCII hex codes.
        self._mapKeys()
        # Convert the help text to ascii codes.
        self._convertHelpText()
        # Read stream and process data.
        self._writeDatatoBufferHelper()
        # Start the actual loop.
        self.startLoop()


    def _initializeMovementVariables(self, *args, **kwargs):
        """
        Sets the initial variables for moving around inside the OpenGL window.
        """
        # Initial x and y coordinates.
        self.x_pos = 0.0
        self.y_pos = 0.0
        # Initial velocities of movement in the x and y direction.
        self.x_vel = 0.0
        self.y_vel = 0.0
        # Initial scale values. Needed for zoom/gain of the Traces.
        self.x_scale = 1.0
        self.y_scale = 1.0

        
    def _setMiscVariables(self, *args, **kwargs):
        """
        Sets variables not fitting for other groups.
        """
        # Initial fps displayed before the first calculation is finished.
        self.fps = '--'
        # Initial time used to calculate the framerate.
        self.fps_time = time()
        # Initial counter value. Every frame will be counted in this variable
        # until one second has passed. Then it will be reset.
        self.fps_counter = 0
        # Determines whether the axis are shown or not.
        self.axis = True
        # Determines whether the help text shows or not.
        self.help = True
        # Number of the glut window.
        self.window = 0


    def _convertHelpText(self):
        """
        OpenGL can only draw chars based on ASCII numbers. This method converts
        the help text to the corresponding ASCII numbers and stores the
        resulting tuple in self.help_text.
        """
        help = ["+/-  zoom", "h/l  left/right", "j/k  up/down",
                "r    reset view", "?    toggle help", "a    toggle axis",
                "u/d  gain/loss", "ESC  exit"]
        temp = []
        for line in help:
            temp_line = []
            for char in line:
                temp_line.append(ord(char))
            temp.append(temp_line)
        # Store in class variable.
        self.help_text = tuple(temp)


    def _mapKeys(self, *args, **kwargs):
        """
        OpenGL only understands octal ASCII key codes.

        These are mapped in this method.
        """
        # Create dictionary.
        self.keys = {}
        # Fill dictionary.
        self.keys['escape'] = '\033'
        self.keys['h'] = '\150'
        self.keys['l'] = '\154'
        self.keys['k'] = '\153'
        self.keys['j'] = '\152'
        self.keys['u'] = '\165'
        self.keys['d'] = '\144'
        self.keys['a'] = '\141'
        self.keys['plus'] = '\053'
        self.keys['minus'] = '\055'
        self.keys['r'] = '\162'
        self.keys['?'] = '\077'

        
    def _writeDatatoBufferHelper(self, *args, **kwargs):
        """
        Reads the file and converts it and writes it to a Vertex Buffer Helper
        object.
        """
        # Read the time exactly once and transform it for faster viewing.
        st = read('test.gse2')
        
        self.starttime = st[0].stats.starttime
        self.endtime = st[0].stats.endtime
        data = st[0].data
        data = data.astype('float32')
        self.mean = data.mean()

        # Remove mean value.
        data = data - self.mean
        self.max_value = data.max()
        self.min_value = data.min()
        self.max_diff = max([self.min_value, self.max_value])
        data = (data / self.max_diff) * 80
        # Read length once for faster access.
        self.length = len(data)
        
        # Create a array with three points for each point.
        data_points = np.empty((self.length, 3), dtype = 'float32')
        # Write x_values
        yy = np.linspace(-80.0, 80.0, self.length) 
        yy = yy.astype('float32')
        
        data_points[:,0] = yy
        # Write y_values
        data_points[:,1] = data
        # Set z-values to 0.
        data_points[:,2] = np.zeros(self.length, dtype = 'float32')
        
        # Now most points need to be doubled.
        temp = np.empty((self.length*2, 3), dtype = 'float32')
        temp[0::2] = data_points
        temp[1::2] = data_points
        # Throw away the first and last value
        data_points = temp[1:-1]
        self.array_length = 2* self.length - 2

        # Read into Buffer helper.
        self.vertex_buffer = vbo.VBO(data_points)


    def InitGL(self):
        """
        Init Function called right after the OpenGL window is created.
        """
        # Set the background color to black.
        glClearColor(0.0, 0.0, 0.0, 0.0)
    
        # XXX: Figure out what the next four calls exactly do and if they are
        # necessary/could be replaced with cheaper calls.
        glClearDepth(1.0)                   # Enables Clearing Of The Depth Buffer
        glDepthFunc(GL_LESS)                # The Type Of Depth Test To Do
        glEnable(GL_DEPTH_TEST)             # Enables Depth Testing
        glShadeModel(GL_SMOOTH)             # Enables Smooth Color Shading
    
        # XXX: Look into Backface Culling and Depth Buffer and optimize if
        # possible.
        
        ###########################
        # SETUP THE PROJETION MATRIX
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Make an orthographic projetion and set a virtual resolution coordinate
        # system going from -100 to 100 in each direction.
        # The last two arguments are the near and far clipping planes.
        # XXX: Deprecated: Replace with glMultMatrix.
        glOrtho(-100.0,100.0,-100.0,100.0,0,128);
        ###########################
    
        # From now on the Matrixmode Matrix will be edited. 
        glMatrixMode(GL_MODELVIEW)
    
    
    def specialKeyPressed(self, key, x, y):
        """
        Handles special key actions, e.g. keys not mappeable via ASCII chars.

        Not implemented yet.
        """
        pass
    
    
    def ReSizeGLScene(self, Width, Height):
        """
        Function called when the Window is Resized.
        """
        # Prevent A Divide By Zero If The Window Is Too Small 
        if Height == 0:
            Height = 1
        # Reset The Current Viewport And Perspective Transformation
        glViewport(0, 0, Width, Height)     
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-100.0,100.0,-100.0,100.0,0,128);
        glMatrixMode(GL_MODELVIEW)
    
    
    def DrawGLScene(self):
        # XXX: Printing something is necessary for updating the window. Why? And
        # how to avoid it??
        print time()
    
        # Clear The Screen And The Depth Buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()                   
    
        # FRAME COUNTER
        # XXX: Has a huge impact on performance. Just for debugging purposes.
        # Measure time.
        self.fps_counter += 1
        if time() - self.fps_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_time = time()
            self.fps_counter = 0
        # Actually draw the frame rate to the window.
        glRasterPos(-99.0, 94.0)
        for char in str(self.fps) + ' FPS':
            glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(char))
    
        # Use the vertex buffer.
        self.vertex_buffer.bind() 
    
        ### Kinetic movement in x- and y direction.
        # Use the velocity to move.
        self.x_pos += self.x_vel
        self.x_vel = 0.99 * self.x_vel
        if self.x_vel < 0.0001 and self.x_vel > 0:
            self.x_vel = 0
        if self.x_vel > -0.0001 and self.x_vel < 0:
            self.x_vel = 0
        # Use the velocity to move.
        self.y_pos += self.y_vel
        self.y_vel = 0.99 * self.y_vel
        if self.y_vel < 0.0001 and self.y_vel > 0:
            self.y_vel = 0
        if self.y_vel > -0.0001 and self.y_vel < 0:
            self.y_vel = 0
            
        # Draw axis.
        if self.axis:
            self.drawAxis()
    
        # Draw Help:
        if self.help:
            self.drawHelp()
    
        # Set the view.
        glTranslatef(self.x_pos * self.x_scale, self.y_pos * self.y_scale, 0.0)
    
        # Use scaling to zoom in.
        glScalef(self.x_scale, self.y_scale, 1.0)
    
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointerf(self.vertex_buffer)
        glDrawArrays(GL_LINES, 0, 2 * self.length)
    
        # Swap the double buffer to avoid ugly half rendered artifacts.
        glutSwapBuffers()

    
    def drawHelp(self):
        """
        Draws the help text.
        """
        start_x = 71.0
        start_y = 94.0
        # Draw stuff
        for _i, line in enumerate(self.help_text):
            y_value = start_y - _i * 8
            glRasterPos(start_x, y_value)
            for char in line:
                glutBitmapCharacter(GLUT_BITMAP_8_BY_13, char)
    
    
    def drawAxis(self):
        """
        Draws the axis.
        """
        ## Just some stuff to figure out positional calculations.
        glColor4f(0.3,0.3,0.3,1.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 200.0, -2.0)
        glVertex3f(0.0, -82.0, -2.0)
        glVertex3f(80.0, 200.0, -2.0)
        glVertex3f(80.0, -82.0, -2.0)
        glEnd()
        glColor4f(1.0,1.0,1.0,1.0,)
    
        cursor_pos = -self.x_pos
        cur_pos = (cursor_pos + 80.0) / 160.0
        cur_pos = self.starttime + (self.endtime - self.starttime) * cur_pos
        cur_pos = cur_pos.strftime('%X')
        glRasterPos(-25, 94.0)
    
        xx = cursor_pos + 80 * 1/self.x_scale
        xx = (xx + 80.0)/160.0
        xx = self.starttime + (self.endtime - self.starttime) * xx
        xx = xx.strftime('%X')
    
        yy = cursor_pos - 80 * 1/self.x_scale
        yy = (yy + 80.0)/160.0
        yy = self.starttime + (self.endtime - self.starttime) *yy
        yy = yy.strftime('%X')
    
        y_value = -80.0
        x_start = -80.0
        x_end = 80.0
        x_range = x_end - x_start
        glBegin(GL_LINES)
        glVertex3f(-81.0, y_value, -1.0)
        glVertex3f(110.0, y_value, -1.0)
        times = [yy, cur_pos, xx]
        # Draw short vertical lines.
        for _i in range(len(times)):
            x_position = x_start + _i * x_range /(len(times) -1)
            glVertex(x_position, y_value -1, -1.0)
            glVertex(x_position, y_value +1, -1.0)
        glEnd()
        # Draw the labels.
        for _i in range(len(times)):
            glRasterPos((x_start-6 * 1/self.x_scale) + _i * x_range /
                        (len(times) - 1), y_value -12)
            for char in times[_i]:
                glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(char))

        x_value = -80.0
        glBegin(GL_LINES)
        glVertex3f(x_value, 100.0, -1.0)
        glVertex3f(x_value, -82.0, -1.0)
        glEnd()
        glColor4f(0.3,0.3,0.3,1.0)
        glBegin(GL_LINES)
        glVertex3f(-80.0, 80.0, -2.0)
        glVertex3f(100.0, 80.0, -2.0)
        glVertex3f(-80.0, 0.0, -2.0)
        glVertex3f(100.0, 0.0, -2.0)
        glEnd()
        glColor4f(1.0, 1.0, 1.0, 1.0)

        data_diff = self.max_value - self.min_value

        middle = (self.mean + (-self.y_pos / self.y_scale) / 80.0 *
                             self.max_diff)

        diff = self.max_diff/self.y_scale

        glRasterPos(-95.0, -2.0, -1.0)
        text = str('%.1f' % middle)
        # XXX: Terribly inefficient way to do it. Once I have access to the
        # internet I need to change it!
        # Probably even better: adjust glRasterPos instead of padding the text!
        if len(text) << 7:
            text = (7 - len(text)) * ' ' + text
        for char in text:
            glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(char))

        glRasterPos(-95.0, 78.0, -1.0)
        text = str('%.1f' % (middle + diff))
        if len(text) << 7:
            text = (7 - len(text)) * ' ' + text
        for char in text:
            glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(char))

        glRasterPos(-95.0, -82.0, -1.0)
        text = str('%.1f' % (middle - diff))
        if len(text) << 7:
            text = (7 - len(text)) * ' ' + text
        for char in text:
            glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(char))

    
    
    def keyPressed(self, *args):
        """
        Handle key presses.
        """
        # If escape is pressed, kill everything.
        if args[0] == self.keys['escape']:
            sys.exit()
    
        # Zoom in and out.
        elif args[0] == self.keys['plus']:
            self.x_scale = 1.05 * self.x_scale
        elif args[0] == self.keys['minus']:
            self.x_scale = 0.95 * self.x_scale

        # Gain/Loss.
        elif args[0] == self.keys['u']:
            self.y_scale = 1.05 * self.y_scale
        elif args[0] == self.keys['d']:
            self.y_scale = 0.95 * self.y_scale
    
        # Move left and right.
        elif args[0] == self.keys['h']:
            if self.x_vel <= 0:
                self.x_vel = 0
            self.x_vel += 0.1
        elif args[0] == self.keys['l']:
            if self.x_vel >= 0:
                self.x_vel = 0
            self.x_vel -= 0.1
           
        # Move up and down.
        elif args[0] == self.keys['j']:
            if self.y_vel <= 0:
                self.y_vel = 0
            self.y_vel += 0.1
        elif args[0] == self.keys['k']:
            if self.y_vel >= 0:
                self.y_vel = 0
            self.y_vel -= 0.1
    
        # Reset the view.
        elif args[0] == self.keys['r']:
            self.x_scale = 1.0
            self.y_scale = 1.0
            self.x_pos = 0
            self.y_pos = 0
            self.x_vel = 0
            self.y_vel = 0

        # Toggle axis.
        elif args[0] == self.keys['a']:
            if self.axis:
                self.axis = False
            else:
                self.axis = True
    
        # Toggle help.
        elif args[0] == self.keys['?']:
            if self.help:
                self.help = False
            else:
                self.help = True
            
    def startLoop(self):
        # For now we just pass glutInit one empty argument. I wasn't sure what
        # should or could be passed in (tuple, list, ...)
        # Once I find out the right stuff based on reading the PyOpenGL source,
        # I'll address this.
        glutInit(sys.argv)
    
        # Select type of Display mode:   
        #  Double buffer 
        #  RGBA color
        # Alpha components supported 
        # Depth buffer
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        
        glutInitWindowSize(1000, 500)
        
        # the window starts at the upper left corner of the screen 
        glutInitWindowPosition(150, 150)
        
        # Okay, like the C version we retain the window id to use when 
        # closing.
        self.window = glutCreateWindow("ObsPy OpenGL Waveform Viewer")
    
        # Register the drawing function with glut, BUT in Python land, at least
        # using PyOpenGL, we need to set the function pointer and invoke a
        # function to actually register the callback, otherwise it would be
        # very much like the C version of the code.    
        glutDisplayFunc(self.DrawGLScene)
        
        # Uncomment this line to get full screen.
        #glutFullScreen()
    
        # When we are doing nothing, redraw the scene.
        glutIdleFunc(self.DrawGLScene)
        
        # Register the function called when our window is resized.
        glutReshapeFunc(self.ReSizeGLScene)
        
        # Register the function called when the keyboard is pressed.  
        glutKeyboardFunc(self.keyPressed)
    
        # Register the function called when special keys (arrows, page down, etc) are
        # pressed.
        glutSpecialFunc(self.specialKeyPressed)
    
        # Initialize our window. 
        self.InitGL()
    
        # Start Event Processing Engine 
        glutMainLoop()



# Print message to console and instance the class which currently automatically
# handles everything.
print "Hit ESC key to quit."
GL = OpenGLViewer()
