.. _utc_date_time:

===========
UTCDateTime
===========

All absolute time values within ObsPy are consistently handled with the
:class:`~obspy.core.utcdatetime.UTCDateTime` class. It is based on a high
precision POSIX timestamp and not the Python datetime class because precision
was an issue.

Initialization
--------------

.. doctest::

   >>> from obspy.core import UTCDateTime
   >>> UTCDateTime("2012-09-07T12:15:00")
   UTCDateTime(2012, 9, 7, 12, 15)
   >>> UTCDateTime(2012, 9, 7, 12, 15, 0)
   UTCDateTime(2012, 9, 7, 12, 15)
   >>> UTCDateTime(1347020100.0)
   UTCDateTime(2012, 9, 7, 12, 15)

In most cases there is no need to worry about timezones, but they are supported: 

.. doctest::

   >>> UTCDateTime("2012-09-07T12:15:00+02:00")
   UTCDateTime(2012, 9, 7, 10, 15)


Attribute Access
----------------

.. doctest::

   >>> time = UTCDateTime("2012-09-07T12:15:00")
   >>> time.year
   2012
   >>> time.julday
   251
   >>> time.timestamp
   1347020100.0
   >>> time.weekday
   4

Handling time differences
-------------------------

.. doctest::

   >>> time = UTCDateTime("2012-09-07T12:15:00")
   >>> print(time + 3600)
   2012-09-07T13:15:00.000000Z
   >>> time2 = UTCDateTime(2012, 1, 1)
   >>> print(time - time2)
   21644100.0


Exercises
---------

* Calculate the number of hours passed since your birth. Optional: Include the correct
  time zone. The current date and time can be obtained with 

.. doctest::

    >>> UTCDateTime()  # doctest: +SKIP

* Get a list of 10 UTCDateTime objects, starting yesterday at 10:00 with a spacing of 90
  minutes. 

* The first session starts at 09:00 and lasts for 3 hours and 15 minutes. Assuming we want
  to have the coffee break 1234 seconds and 5 microseconds before it ends. At what time is
  the coffee break?

* Assume you had your last cup of coffee yesterday at breakfast. How many minutes do you
  have to survive with that cup of coffee?
