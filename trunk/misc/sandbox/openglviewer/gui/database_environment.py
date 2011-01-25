#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from time import time

class DatabaseEnvironment(object):
    """
    Super class that stores any variables necessary for all other functions.

    Also used to store global variables since they are available to all other
    methods.
    """
    def __init__(self, *args, **kwargs):
        self.errorHandler = ErrorHandler()
        # Some globally available variables.
        self.seishub_server = 'http://localhost:7777'
        self.seishub_ping_interval = 1
        self.cache_dir = 'cache'
        self.debug = kwargs.get('debug', False)

class Error(object):
    """
    Every error gets stored in that class.
    """
    def __init__(self, msg):
        """
        Just stores some errors.
        """
        self.timestamp = time()
        self.msg = msg

class CaughtException(Exception):
    """
    This error is used for Exceptions that will be handled by the error
    handler.
    """
    pass

class ErrorHandler(object):
    """
    All errors will be handled and printed by this class.
    """

    def __init__(self):
        self.errors = []

    def addError(self, msg):
        """
        Appends an error object to self.errors.
        """
        self.erros.append(Error(msg))
