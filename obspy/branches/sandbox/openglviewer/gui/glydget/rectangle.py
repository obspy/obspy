#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# glydget - an OpenGL widget toolkit
# Copyright (c) 2009 - Nicolas P. Rougier
#
# This file is part of glydget.
#
# glydget is free software: you can  redistribute it and/or modify it under the
# terms of  the GNU General  Public License as  published by the  Free Software
# Foundation, either  version 3 of the  License, or (at your  option) any later
# version.
#
# glydget is  distributed in the hope that  it will be useful,  but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy  of the GNU General Public License along with
# glydget. If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
''' Rectangle object. '''
import pyglet
import pyglet.gl as gl


class Rectangle(object):
    ''' Rectangle object. '''

    def __init__(self, x=0, y=0, width=0, height=0, colors=(0,0,0,100), filled=True):
        ''' Create a rectangle.

        :Parameters:
            `x` : int
                X coordinate ot the top-left corner.
            `y` : int
                Y coordinate ot the top-left corner.
            `width` : int
                Width of the object in pixels.
            `height` : int
                Height of the object in pixels
            `colors`: (int,int,int,int) or [int,int,int,int] *4 or *8
                4-tuples of RGBA components or flat list of 4 or 8 RGBA components.
            `filled` : bool
                Indicate whether rectangle is filled or not.
        '''
        self._update_enabled = False
        self._vertices = None
        self._batch = None
        self._group = None
        self._position = [x,y]
        self._size = [width,height]
        self._filled = filled
        self.colors = colors
        self.build()


    def build(self, batch=None, group=None):
        ''' Build rectangle and add it to batch and group.

        :Parameters:
            `batch` : `Batch`
                Optional graphics batch to add the object to.
            `group` : `Group`
                Optional graphics group to use.
        '''

        self._batch = batch or self._batch or pyglet.graphics.Batch()
        self._group = group or self._group

        if self._colors[3] == 0:
            return

        if self._vertices:
            self.delete()
        x,y = int(self._position[0]), int(self._position[1])
        w,h = int(self._size[0]), int(self._size[1])

        if self._filled:
            self._vertices = self._batch.add(4, gl.GL_QUADS, self._group,
                                             ('v2f', (x,y,x+w,y,x+w,y-h,x,y-h)),
                                             ('c4B', self._colors))
        else:
            x,y = x+.315, y-.315
            w,h = w-1, h-1
            self._vertices = self._batch.add(8, gl.GL_LINES, self._group,
                                             ('v2f', (x,y,x+w,y,x+w,y,x+w,y-h,
                                                      x+w,y-h,x,y-h,x,y-h,x,y)),
                                             ('c4B', self._colors))

    def delete(self):
        ''' Delete rectangle.  '''
        if not self._vertices:
            return
        self._vertices.delete()
        self._vertices = None



    def move(self, x, y):
        ''' Move object to (x,y).

        :Parameters:
            `x` : int
                X coordinate ot the top-left corner
            `y` : int
                Y coordinate ot the top-left corner
        '''
        #self.begin_update()
        self._position = [x,y]
        #self.end_update()



    def resize(self, width, height):
        ''' Resize object to (width, height).

        :Parameters:
            `width` : int
                Width of the object in pixels
            `height` : int
                Height of the object in pixels
        '''
        #self.begin_update()
        self._size = [width, height]
        #self.end_update()



    def begin_update(self):
        '''Indicate that a number of changes to the object are about
        to occur.

        Changes to the size or position between calls to `begin_update` and
        `end_update` do not trigger any costly operations. All changes are
        performed when `end_update` is called.
        '''
        self._update_enabled = False



    def end_update(self):
        '''Perform pending changes since `begin_update`.

        See `begin_update`.
        '''
        self._update_enabled = True
        self._update()



    def _update(self):
        '''Update size and position of the object.

        See `begin_update` and `end_update`.
        '''
        if not self._vertices:
            return
        x,y = int(self._position[0]), int(self._position[1])
        w,h = int(self._size[0]), int(self._size[1])
        if self._filled:
            self._vertices.vertices = [x,y,x+w,y,x+w,y-h,x,y-h]
        else:
            x,y = x+.315, y-.315
            w,h = w-1, h-1
            self._vertices.vertices = [x,y,x+w,y,x+w,y,x+w,y-h,
                                       x+w,y-h,x,y-h,x,y-h,x,y]
        self._vertices.colors = self._colors


    def _update_colors(self):
        '''Update rectangle colors.        
        '''
        #if not self._vertices:
        #    return
        #self._vertices.colors = self._colors
        if self._colors[3] == 0:
            self.delete()
        else:
            self.build()
        

       
    def _get_colors(self):
        return self._colors

    def _set_colors(self, colors):
        self.begin_update()
        if type(colors) == list:
            if self._filled:
                assert(len(colors) == 16)
            else:
                assert(len(colors) == 32)
            self._colors = colors
        elif type(colors) == tuple:
            if self._filled:
                self._colors = colors*4
            else:
                self._colors = colors*8
        else:
            raise ValueError, \
                '''colors must be a 4-tuple of RGBA components or a flat list'''\
                '''of 4 or 8 RGBA components.'''
        self.end_update()

    colors = property(_get_colors, _set_colors,
                     doc='''Object color.

    For filled rectangle, colors is a flat list of 4 RGBA components,
    each in range [0, 255].

    For outlined rectangle, colors is a flat list of 8 RGBA components,
    each in range [0, 255].

    :type: (int,int,int,int) or [int,int,int,int] *4 or *8
    ''')



    def _get_x(self):
        return self._position[0]

    def _set_x(self, x):
        self.begin_update()
        self._position = [x, self._position[1]]
        self.end_update()

    x = property(_get_x, _set_x,
                 doc='''X coordinate of top-left corner.
    
    :type: int
    ''')



    def _get_y(self):
        return self._position[1]

    def _set_y(self, y):
        self.begin_update()
        self._position = [self._position[0], y]
        self.end_update()

    y = property(_get_y, _set_y,
                 doc='''Y coordinate of top-left corner.
    
    :type: int
    ''')



    def _get_width(self):
        return self._size[0]

    def _set_width(self, width):
        self.begin_update()
        self._size = [width, self._size[1]]
        self.end_update()

    width = property(_get_width, _set_width,
                     doc='''Rectangle width in pixels.

    :type: int
    ''')



    def _get_height(self):
        return self._size[1]

    def _set_height(self, height):
        self.begin_update()
        self._size = [self._size[0], height]
        self.end_update()

    height = property(_get_height, _set_height,
                      doc='''Rectangle height in pixels.

    :type: int
    ''')



    def _get_batch(self):
        return self._batch

    def _set_batch(self, batch):
        self.build(batch, self._group)

    batch = property(_get_batch, _set_batch,
                      doc='''Graphics batch where rectangle is currently in.

    :type: pyglet.graphics.Batch
    ''')



    def _get_group(self):
        return self._group

    def _set_group(self, group):
        self.build(self._batch, group)

    group = property(_get_group, _set_group,
                      doc='''Graphics group this rectangle belongs.

    :type: pyglet.graphics.Group
    ''')





if __name__ == '__main__':
    import random
    pyglet.options['debug_gl'] = False
    import pyglet.gl as gl

    window = pyglet.window.Window(512, 512, vsync=0)
    fps_display = pyglet.clock.ClockDisplay()
    pyglet.clock.schedule(lambda dt: None)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    batch = pyglet.graphics.Batch()
    bg_group = pyglet.graphics.OrderedGroup(0)
    fg_group = pyglet.graphics.OrderedGroup(1)

    for i in range(500):
        w,h = 100, 20
        x = random.randint(-w//2,window.width-w//2)
        y = random.randint(h//2,window.height+h//2)
        r,g,b,a = (random.randint(0,255), random.randint(0,255),
                   random.randint(0,255), random.randint(0,255)) 
        rect = Rectangle(x,y,w,h,(r,g,b,a),True)
        rect.build(batch,bg_group)
        rect = Rectangle(x,y,w,h,(r,g,b,0),False)
        rect.build(batch,fg_group)

    @window.event
    def on_draw():
        gl.glClearColor(0,0,0,1)
        window.clear()
        batch.draw()
        fps_display.draw()
    pyglet.app.run()

