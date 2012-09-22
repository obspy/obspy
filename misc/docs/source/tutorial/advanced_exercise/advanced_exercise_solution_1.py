import obspy.neries

client = obspy.neries.Client()

events = client.getEvents(min_latitude=46.1, max_latitude=46.3,
        min_longitude=7.6, max_longitude=7.8,
        min_datetime="2012-04-03", max_datetime="2012-04-04")

print "found %s event(s):" % len(events)
for event in events:
    print event
