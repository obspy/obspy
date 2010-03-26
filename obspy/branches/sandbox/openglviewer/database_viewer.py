#1!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import pyglet and do not debug the OpenGL code.
import pyglet
pyglet.options['debug_gl'] = False
# Import custom made GUI stuff.
from gui import *

if __name__ == '__main__':
    """
    Just testing some stuff.
    """
    # Path to the previews.
    preview_path = 'preview/'

    # Launch the globally available Environment class which later needs to be
    # given as an argument to most other class instances.
    env = DatabaseEnvironment(debug = True)
        
    # Set the window.
    win = PygletWindow(env = env)

    # Shortcut to window because it gets called frequently.
    window = win.window

    # Handle all the loops.
    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        """
        Handle mouse scrolls.

        XXX: Need way better implementation.
        """
        win.mouse_scroll(x, y, scroll_x, scroll_y)

    @window.event
    def on_draw(*args, **kwargs):
        """
        The actual main loop. Do no intensive calculations in there.
        """
        win.draw()
        # XXX: Need to figure out a way to add the menu to the batch.
        win.menu.menu.batch.draw()

    @window.event
    def on_resize(width, height):
        """
        Gets called on resize.
        """
        win.resize(width, height)

    # Kick off the mainloop.
    pyglet.app.run()
