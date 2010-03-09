#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time
from utils import get_channels

class DatabaseEnvironment(object):
    """
    Super class that stores any variables necessary for all other functions.

    Also used to store global variables since they are available to all other
    methods.
    """
    def __init__(self):
        self.errorHandler = ErrorHandler()

    def setPath(self, base_path):
        self.path = base_path
        self.paths = get_channels(base_path)
            


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
