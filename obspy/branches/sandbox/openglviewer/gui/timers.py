#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gui_element import GUIElement
import pyglet

class Timers(GUIElement):
    """
    Collects all kind of functions that will be called periodically.
    """
    def __init__(self, *args, **kwargs):
        """
        Usual init method.
        """
        super(Timers, self).__init__(self, **kwargs)
        self.pingServer(0)
        pyglet.clock.schedule_interval(self.pingServer, self.env.seishub_ping_interval)

    def pingServer(self, dt):
        """
        Pings the server.
        """
        try:
            ping = self.win.seishub.ping()
            if ping:
                msg = "SeisHub server: %s" % self.env.seishub_server
                self.win.status_bar.setServer(msg)
            else:
                msg = "SeisHub server %s not reacheable." % self.env.seishub_server
                self.win.status_bar.setServer(msg)

        except:
            msg = "SeisHub server %s not reacheable." % self.env.seishub_server
            self.win.status_bar.setServer(msg)

