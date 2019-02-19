==========================
Cross Correlation Detector
==========================

These code snippets shows how to use the functions
:func:`~obspy.signal.cross_correlation.correlate_stream_template`,
:func:`~obspy.signal.cross_correlation.similarity_detector` and
:func:`~obspy.signal.cross_correlation.insert_amplitude_ratio`.

--------------------------------
Detection based on one component
--------------------------------

In the first example we will determine the origin time of the 2017
North Korean nuclear test, by using a template of another test in 2013. We will
use only a single channel of the station IC.MDJ.


.. plot::
    :context: reset
    :include-source:

    from obspy import read, UTCDateTime as UTC
    from obspy.signal.cross_correlation import correlate_stream_template, similarity_detector

    template = read('https://examples.obspy.org/IC.MDJ.2013.043.mseed')
    template.filter('bandpass', freqmin=0.5, freqmax=2)
    template.plot()

.. plot::
    :context: close-figs
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
    detections


.. testoutput::

    [{'time': 2017-09-03T03:30:01.371731Z, 'similarity': 0.46020218652131661},
     {'time': 2017-09-03T03:38:31.821731Z, 'similarity': 0.21099726068286467}]


--------------------------------------------
Multi-station detection of swarm earthquakes
--------------------------------------------

In this example we load 12 hours of data from the start of the 2018 Novy Kostel
earthquake swarm in Northwestern Bohemia/Czech Republic near the border to Germany.
The example stream consists only of Z component data.
Origin time and magnitude of the largest earthquake in this period are
extracted from the WEBNET earthquake catalog.
Data are filtered by a highpass and the template waveforms of this earthquake are selected and plotted.
After that, cross correlations are caluclated and other earthquakes in the swarm are detected.

.. plot::
    :context: reset
    :include-source:

    from obspy import read, Trace, UTCDateTime as UTC
    from obspy.signal.cross_correlation import correlate_stream_template, insert_amplitude_ratio, similarity_detector

    stream = read('https://examples.obspy.org/NKC_PLN_ROHR.HHZ.2018.130.mseed')
    stream.filter('highpass', freq=1, zerophase=True)
    otime = UTC('2018-05-10 14:24:50')
    template = stream.select(station='NKC').slice(otime + 2, otime + 7)
    template += stream.select(station='ROHR').slice(otime + 2, otime + 7)
    template += stream.select(station='PLN').slice(otime + 6, otime + 12)
    template.plot()


.. plot::
    :context: close-figs
    :include-source:

    ccs = correlate_stream_template(stream, template, template_time=otime)
    detections = similarity_detector(ccs, 0.5, 10, 10, plot_detections=stream)

Note, that the stream of cross correlations in the variable ccs is also suitable for use with
:func:`~obspy.signal.trigger.coincidence_trigger`, but that function will return the trigger time,
when we are interessed in the time when the similarity is maximized.

In the following, we create the similarity trace on our own and introduce the
constraint that the cross correlation should be larger than 0.5 at all stations.


.. plot::
    :context: close-figs
    :include-source:

    def similarity_component_thres(ccs, thres, num_components):
        """Return Trace with mean of ccs
        and set values to zero if number of components above thresshold is not reached"""
        ccmatrix = np.array([tr.data for tr in ccs])
        header = dict(sampling_rate=ccs[0].stats.sampling_rate,
                      starttime=ccs[0].stats.starttime)
        comp_thres = np.sum(ccmatrix > thres, axis=0) >= num_components
        data = np.mean(ccmatrix, axis=0) * comp_thres
        return Trace(data=data, header=header)

    similarity = similarity_component_thres(ccs, 0.5, 3)
    detections = similarity_detector(None, 0.5, 10, 10, similarity=similarity, plot_detections=stream)

Now, we have only 7 detections, probably from a specific earthquake cluster.
To get more detections, we need to relax the constraints again.
Another possibility is to calculate the envelope of the data before applying the correlation.

Finally, amplitude ratios between the detections and the template and magnitude
estimates based on the amplitude ratio and the magnitude of the template event
are inserted into the detection list.

.. testsetup::

    #prep from above
    from obspy import read, Trace, UTCDateTime as UTC
    from obspy.signal.cross_correlation import correlate_stream_template, insert_amplitude_ratio, similarity_detector

    stream = read('https://examples.obspy.org/NKC_PLN_ROHR.HHZ.2018.130.mseed')
    stream.filter('highpass', freq=1, zerophase=True)
    otime = UTC('2018-05-10 14:24:50')
    template = stream.select(station='NKC').slice(otime + 2, otime + 7)
    template += stream.select(station='ROHR').slice(otime + 2, otime + 7)
    template += stream.select(station='PLN').slice(otime + 6, otime + 12)

    def similarity_component_thres(ccs, thres, num_components):
        """Return Trace with mean of ccs
        and set values to zero if number of components above thresshold is not reached"""
        ccmatrix = np.array([tr.data for tr in ccs])
        header = dict(sampling_rate=ccs[0].stats.sampling_rate,
                      starttime=ccs[0].stats.starttime)
        comp_thres = np.sum(ccmatrix > thres, axis=0) >= num_components
        data = np.mean(ccmatrix, axis=0) * comp_thres
        return Trace(data=data, header=header)

    similarity = similarity_component_thres(ccs, 0.5, 3)
    detections = similarity_detector(None, 0.5, 10, 10, similarity=similarity)



.. testcode::

    insert_amplitude_ratio(detections, stream, template, template_time=otime, template_magnitude=2.9)


.. testoutput::

    [{'time': 2018-05-10T12:34:56.630000Z,
      'similarity': 0.7248917248719996,
      'amplitude_ratio': 0.042826872986209588,
      'magnitude': 1.0756218205928332},
     {'time': 2018-05-10T14:24:50.000000Z,
      'similarity': 0.99999999999999967,
      'amplitude_ratio': 1.0,
      'magnitude': 2.8999999999999999},
     {'time': 2018-05-10T14:27:50.920000Z,
      'similarity': 0.57155043392492477,
      'amplitude_ratio': 0.019130460518598909,
      'magnitude': 0.60896723296053024},
     {'time': 2018-05-10T14:41:07.690000Z,
      'similarity': 0.77287907439378944,
      'amplitude_ratio': 0.57507924545222067,
      'magnitude': 2.5796369256528813},
     {'time': 2018-05-10T14:55:50.000000Z,
      'similarity': 0.57467717600498891,
      'amplitude_ratio': 0.078631249252299668,
      'magnitude': 1.4274602340872211},
     {'time': 2018-05-10T15:12:10.140000Z,
      'similarity': 0.68520826878360419,
      'amplitude_ratio': 0.11301513001944399,
      'magnitude': 1.6375154520085005},
     {'time': 2018-05-10T19:22:29.510000Z,
      'similarity': 0.70112087830579517,
      'amplitude_ratio': 0.68929540439903225,
      'magnitude': 2.6845405106924867}]
