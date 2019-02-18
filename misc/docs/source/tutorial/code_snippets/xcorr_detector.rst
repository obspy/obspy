==========================
Cross Correlation Detector
==========================

This code snippets shows how to use the functions
:func:`~obspy.signal.cross_correlation.correlate_stream_template` and
:func:`~obspy.signal.cross_correlation.similarity_detector`.

In the first example we will determine the origin time of the 2017
North Korean nuclear test, by using a template of another test in 2013. We will
use only a single channel of the station IC.MDJ.


.. plot::
    :context:
    :include-source:

    from obspy import read, UTCDateTime as UTC
    from obspy.signal.cross_correlation import correlate_stream_template, similarity_detector

    template = read('https://examples.obspy.org/IC.MDJ.2013.043.mseed')
    template.filter('bandpass', freqmin=0.5, freqmax=2)
    template.plot()

.. plot::
    :context:
    :include-source:

    pick = UTC('2013-02-12T02:58:44.95')
    template.trim(pick, pick + 150)

    stream = read('https://examples.obspy.org/IC.MDJ.2017.246.mseed')
    stream.filter('bandpass', freqmin=0.5, freqmax=2)
    ccs = correlate_stream_template(stream, template)
    detections = similarity_detector(ccs, 0.3, 10, 10, plot_detections=stream)


The above detection corresponds to the starttime of the template.
If the template of the 2013 explosion is associated with its origin time,
the origin time of the 2017 explosion can be directly determined.
The thresshold is lowered to 0.2 to detect also the collapse which occured
around 8 minutes after the 2013 test.

.. testsetup::

    #prep from above
    from obspy import read, UTCDateTime as UTC
    from obspy.signal.cross_correlation import correlate_stream_template, similarity_detector
    template = read('https://examples.obspy.org/IC.MDJ.2013.043.mseed')
    template.filter('bandpass', freqmin=0.5, freqmax=2)
    pick = UTC('2013-02-12T02:58:44.95')
    template.trim(pick, pick + 150)
    stream = read('https://examples.obspy.org/IC.MDJ.2017.246.mseed')
    stream.filter('bandpass', freqmin=0.5, freqmax=2)


.. testcode::

    utc_nuclear_test_2013 = UTC('2013-02-12T02:57:51')
    ccs = correlate_stream_template(stream, template, template_time=utc_nuclear_test_2013)
    detections = similarity_detector(ccs, 0.2, 10, 10)
    print('number of detections:', len(detections))
    print('detections:', ', '.join(str(d) for d in detections))


.. testoutput::

    number of detections: 2
    detections: 2017-09-03T03:30:01.371731Z, 2017-09-03T03:38:31.821731Z

