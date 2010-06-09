from PyQt4 import QtCore
from obspy.core import UTCDateTime

MONTHS = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul',
          8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dez'}

def toQDateTime(dt):
    """
    Converts a UTCDateTime object to a QDateTime object.
    """
    return QtCore.QDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                        dt.second, dt.microsecond//1000, QtCore.Qt.TimeSpec(1))

def fromQDateTime(dt):
    """
    Converts a QDateTime to a UTCDateTime object.
    """
    # XXX: Microseconds might be lost, but that does not matter for the
    # purpose of the database viewer.
    return UTCDateTime(dt.toPyDateTime())
