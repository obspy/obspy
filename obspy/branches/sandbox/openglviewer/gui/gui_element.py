#!/usr/bin/env python
# -*- coding: utf-8 -*-

class GUIElement(object):
    """
    Class that all GUI Elements should inherit from.
    """
    def __init__(self, *args, **kwargs):
        """
        Just some variables needed for every GUI element.
        """
        parent = kwargs.get('parent')
        # Environment.
        self.env = parent.env
        # Every gui element needs a parent window.
        self.win = parent.win
        # In the end there will be only one single batch. This construct here
        # assures that it will always get to the deepest batch.
        self.batch = parent.batch
        # Set the group. If none is given just draw on top of everything else.
        self.group = kwargs.get('group', 999)
        if type(self.group) == int:
            self.group = self.win.getGroup(self.group)

    def resize(self, width, height):
        """
        Dummy resize method each GUI Element should have.

        If none is present nothing will happen upon resize.

        Get the window porperties from self.windows.
        """
        pass
