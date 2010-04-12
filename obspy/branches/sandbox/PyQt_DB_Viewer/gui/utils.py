from PyQt4 import QtCore
from obspy.core import UTCDateTime

def toQDateTime(dt):
    """
    Converts a UTCDateTime object to a QDateTime object.
    """
    return QtCore.QDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                            dt.second, dt.microsecond, QtCore.Qt.TimeSpec(1))

def fromQDateTime(dt):
    """
    Converts a QDateTime to a UTCDateTime object.
    """
    # XXX: Microseconds might be lost, but that does not matter for the
    # purpose of the database viewer.
    return UTCDateTime(dt.toPyDateTime())
