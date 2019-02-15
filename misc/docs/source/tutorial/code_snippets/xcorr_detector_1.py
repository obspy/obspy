from obspy import read, UTCDateTime as UTC
from obspy.signal.cross_correlation import (correlate_stream_template,
                                            similarity_detector)

template = read('"https://examples.obspy.org/IC.MDJ.2013.043.mseed')
template.filter('highpass', freq=1)
template.plot()
pick = UTC('2013-02-12T02:58:44.95')
template.trim(pick, pick + 150)

stream = read('"https://examples.obspy.org/IC.MDJ.2017.246.mseed')
stream.filter('highpass', freq=1)
ccs = correlate_stream_template(stream, template)
detections = similarity_detector(ccs, 0.3, 10, 10, plot_detections=stream)
