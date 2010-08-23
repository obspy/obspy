# -*- coding: utf-8 -*-

from util import UniqueList, raise_locked_warning

class Event(object):
    """
    Class that handles picks and location for events.
    """
    def __init__(self):
        # Status variable that determine whether the object and its picks can
        # be edited.
        # The base class needs to be called explicitly.
        object.__setattr__(self, '_Event__locked', False)

        # All picks will be stored in a unique list to avoid duplicate entries.
        # It also provides some convenience functions.
        self.picks = UniqueList()

        # Sets some common attributes meaningful for all events.
        self._setImportantAttributes()

    def __setattr__(self, name, value):
        """
        Attribute access only when the object is not locked.
        """
        if self.__locked:
            raise_locked_warning()
            return
        # Call the base class and set the value.
        object.__setattr__(self, name, value)

    def _setImportantAttributes(self):
        """
        Sets some important attributes that are meaningful for all events.
        """
        self.origin_time = None
        self.origin_time_error = None
        self.origin_latitude = None
        self.origin_latitude_error = None
        self.origin_longitude = None
        self.origin_longitude_error = None
        self.origin_magnitude = None
        self.origin_magnitude_error = None
        self.origin_magnitude_type = None

    def _lock(self):
        """
        Locks the object and all subobjects.
        """
        self.__locked = True
        # Lock every pick.
        for pick in self.picks:
            pick._lock()
        # Lock the pick list.
        self.picks._lock()

    def _unlock(self):
        """
        Unlocks the object.
        """
        # The base class needs to be called explicitly.
        object.__setattr__(self, '_Event__locked', False)
        # Unlock every pick.
        for pick in self.picks:
            pick._unlock()
        # Lock the pick list.
        self.picks._unlock()

    def locate(self):
        """
        Dummy function that currently just prevents any further changes to the
        object event and all picks.
        """
        self._lock()
