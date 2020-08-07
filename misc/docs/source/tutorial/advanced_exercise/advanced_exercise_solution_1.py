from __future__ import print_function

import obspy
import obspy.clients.fdsn


client = obspy.clients.fdsn.Client("EMSC")

events = client.get_events(minlatitude=46.1, maxlatitude=46.3,
                           minlongitude=7.6, maxlongitude=7.8,
                           starttime=obspy.UTCDateTime("2012-04-03"),
                           endtime=obspy.UTCDateTime("2012-04-04"))

print("found %s event(s):" % len(events))
for event in events:
    print(event)
