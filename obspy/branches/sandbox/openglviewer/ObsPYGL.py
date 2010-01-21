#!/usr/bin/env python

# This is statement is required by the build system to query build info
if __name__ == '__build__':
    raise Exception

import time
from obspy.core import read
import numpy as np

#The OpenGL.arrays.vbo.VBO class is a convenience wrapper 
#which makes it easier to use Vertex Buffer Objects from within 
#PyOpenGL.  It takes care of determining which implementation 
#to use, the creation of offset objects, and even basic slice-based 
#updating of the content in the VBO.
from OpenGL.arrays import vbo


# Read the time exactly once and transform it for faster viewing.
st = read('test.gse2')
data = st[0].data
data = data.astype('float32')
# Make data go from 0 to one 1 horizontally and from 0 to 0.4 vertical.
data = data - data.min()
data = (data / data.max()) * 0.4
# Read length once for faster access.
length = len(data)

# Create a array with three points for each point.
data_points = np.empty((length, 3), dtype = 'float32')
# Write x_values
yy = np.linspace(0.0, 1.0, length) 
yy = yy.astype('float32')

data_points[:,0] = yy
# Write y_values
data_points[:,1] = data
# Set z-values to 0.
data_points[:,2] = np.zeros(length, dtype = 'float32')

# Now most points need to be doubled.
temp = np.empty((length*2, 3), dtype = 'float32')
temp[0::2] = data_points
temp[1::2] = data_points
# Throw away the first and last value
data_points = temp[1:-1]
length = 2* length - 2


# Read into Buffer helper.
vertex_buffer = vbo.VBO(data_points)


# Global variables for Moving the camera.
x_pos = -0.5
y_pos = -0.15
z_pos = -1.0

# Camera velocities used for kinetic movement.
x_vel = 0
y_vel = 0
z_vel = 0

first = True

old_time = time.time()


from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys


# Some api in the chain is translating the keystrokes to this octal string
# so instead of saying: ESCAPE = 27, we use the following.
ESCAPE = '\033'
LEFT_ARROW = '\150'
RIGHT_ARROW = '\154'
UP_ARROW = '\153'
DOWN_ARROW = '\152'
PLUS = '\053'
MINUS = '\055'

# Number of the glut window.
window = 0


# A general OpenGL initialization function.  Sets all of the initial parameters. 
def InitGL(Width, Height):              # We call this right after our OpenGL window is created.
    glClearColor(0.0, 0.0, 0.0, 0.0)    # This Will Clear The Background Color To Black
    glClearDepth(1.0)                   # Enables Clearing Of The Depth Buffer
    glDepthFunc(GL_LESS)                # The Type Of Depth Test To Do
    glEnable(GL_DEPTH_TEST)             # Enables Depth Testing
    glShadeModel(GL_SMOOTH)             # Enables Smooth Color Shading
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()                    # Reset The Projection Matrix
                                        # Calculate The Aspect Ratio Of The Window
    gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)
    glutMouseFunc(testing)

def specialKeyPressed(key, x, y):
    pass

def testing(*args):
    print args

# The function called when our window is resized (which shouldn't happen if you enable fullscreen, below)
def ReSizeGLScene(Width, Height):
    if Height == 0:                     # Prevent A Divide By Zero If The Window Is Too Small 
        Height = 1

    glViewport(0, 0, Width, Height)     # Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def DrawGLScene():
    global first, x_pos, x_vel, z_vel, z_pos, y_vel, y_pos, old_time
    # Measure time.
    xx = time.time()
    fps = 1.0/(xx - old_time)
    old_time = xx
    print '%f FPS' % fps
    # Use the vertex buffer.
    vertex_buffer.bind() 
    # Clear The Screen And The Depth Buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()                   

    # Use the velocity to move.
    z_pos += z_vel
    z_vel = 0.99 * z_vel
    if z_vel < 0.0001 and z_vel > 0:
        z_vel = 0
    if z_vel > -0.0001 and z_vel < 0:
        z_vel = 0
    # Avoid clipping.
    if z_pos >= -0.1:
        z_pos = -0.1

    # Use the velocity to move.
    x_pos += x_vel
    x_vel = 0.99 * x_vel
    if x_vel < 0.0001 and x_vel > 0:
        x_vel = 0
    if x_vel > -0.0001 and x_vel < 0:
        x_vel = 0

    # Use the velocity to move.
    y_pos += y_vel
    y_vel = 0.99 * y_vel
    if y_vel < 0.0001 and y_vel > 0:
        y_vel = 0
    if y_vel > -0.0001 and y_vel < 0:
        y_vel = 0

    # Set the view.
    glTranslatef(x_pos, y_pos, z_pos)

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointerf(vertex_buffer)
    glDrawArrays(GL_LINES, 0, length)

    # Unbind to draw all the extra stuff.
    vertex_buffer.unbind()

    #if first:
    #    glBegin(GL_LINES)                 # Start drawing a polygon
    #    for _i, point in enumerate(data[:-1]):
    #        x_value = float(_i)/length
    #        # Draw a line for each point.
    #        glVertex3f(x_value, data[_i] , 0.0)
    #        glVertex3f(x_value + 1.0/length, data[_i + 1] , 0.0)
    #    
    #    glEnd()                             # We are done with the polygon
    #    first = False

    #  since this is double buffered, swap the buffers to display what just got drawn. 
    glutSwapBuffers()

def keyPressed(*args):
    global x_pos, y_pos, z_pos, z_vel, x_vel, y_vel

    x = args[1]
    y = args[2]
    print args, x,y

    # If escape is pressed, kill everything.
    if args[0] == ESCAPE:
        sys.exit()

    elif args[0] == PLUS:
        # Adjust the z-Movement depending on how close to 0.0 one is.
        z_vel = 0.005

    elif args[0] == MINUS:
        z_vel = -0.005

    elif args[0] == LEFT_ARROW:
        x_vel = +.001

    elif args[0] == RIGHT_ARROW:
        x_vel = -0.001
       
    elif args[0] == UP_ARROW:
        y_vel = -0.001

    elif args[0] == DOWN_ARROW:
        y_vel = +0.001

def main():
    global window
    # For now we just pass glutInit one empty argument. I wasn't sure what should or could be passed in (tuple, list, ...)
    # Once I find out the right stuff based on reading the PyOpenGL source, I'll address this.
    glutInit(sys.argv)

    # Select type of Display mode:   
    #  Double buffer 
    #  RGBA color
    # Alpha components supported 
    # Depth buffer
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    
    # get a 640 x 480 window 
    glutInitWindowSize(1000, 500)
    
    # the window starts at the upper left corner of the screen 
    glutInitWindowPosition(0, 0)
    
    # Okay, like the C version we retain the window id to use when closing, but for those of you new
    # to Python (like myself), remember this assignment would make the variable local and not global
    # if it weren't for the global declaration at the start of main.
    window = glutCreateWindow("ObsPy OpenGL Waveform Viewer")

    # Register the drawing function with glut, BUT in Python land, at least using PyOpenGL, we need to
    # set the function pointer and invoke a function to actually register the callback, otherwise it
    # would be very much like the C version of the code.    
    glutDisplayFunc(DrawGLScene)
    
    # Uncomment this line to get full screen.
    #glutFullScreen()

    # When we are doing nothing, redraw the scene.
    glutIdleFunc(DrawGLScene)
    
    # Register the function called when our window is resized.
    glutReshapeFunc(ReSizeGLScene)
    
    # Register the function called when the keyboard is pressed.  
    glutKeyboardFunc(keyPressed)

    # Register the function called when special keys (arrows, page down, etc) are
    # pressed.
    glutSpecialFunc(specialKeyPressed)

    # Initialize our window. 
    InitGL(1000, 500)

    # Start Event Processing Engine 
    glutMainLoop()

# Print message to console, and kick off the main to get it rolling.
print "Hit ESC key to quit."


main()
