====================================
Clone an Existing Dataless SEED File
====================================

The following code example shows how to clone an existing DatalessSEED file
(``dataless.seed.BW_RNON``) and use it as a template to build up a DatalessSEED
file for a new station.

First of all, we have to make the necessary imports and read the existing
DatalessSEED volume (stored on our `examples webserver`_):

.. doctest::

    >>> from obspy import UTCDateTime
    >>> from obspy.io.xseed import Parser
    >>>
    >>> p = Parser("https://examples.obspy.org/dataless.seed.BW_RNON")
    >>> blk = p.blockettes

Now we can adapt the information only appearing once in the DatalessSEED at the
start of the file, in this case Blockette 50 and the abbreviations in Blockette
33:

.. doctest::

    >>> blk[50][0].network_code = 'BW'
    >>> blk[50][0].station_call_letters = 'RMOA'
    >>> blk[50][0].site_name = "Moar Alm, Bavaria, BW-Net"
    >>> blk[50][0].latitude = 47.761658
    >>> blk[50][0].longitude = 12.864466
    >>> blk[50][0].elevation = 815.0
    >>> blk[50][0].start_effective_date = UTCDateTime("2006-07-18T00:00:00.000000Z")
    >>> blk[50][0].end_effective_date = ""
    >>> blk[33][1].abbreviation_description = "Lennartz LE-3D/1 seismometer"

After that we have to change the information for all of the three channels
involved:

.. doctest::

    >>> mult = len(blk[58])/3
    >>> for i, cha in enumerate(['Z', 'N', 'E']):
    ...     blk[52][i].channel_identifier = 'EH%s' % cha
    ...     blk[52][i].location_identifier = ''
    ...     blk[52][i].latitude = blk[50][0].latitude
    ...     blk[52][i].longitude = blk[50][0].longitude
    ...     blk[52][i].elevation = blk[50][0].elevation
    ...     blk[52][i].start_date = blk[50][0].start_effective_date
    ...     blk[52][i].end_date = blk[50][0].end_effective_date
    ...     blk[53][i].number_of_complex_poles = 3
    ...     blk[53][i].real_pole = [-4.444, -4.444, -1.083]
    ...     blk[53][i].imaginary_pole = [+4.444, -4.444, +0.0]
    ...     blk[53][i].real_pole_error = [0, 0, 0]
    ...     blk[53][i].imaginary_pole_error = [0, 0, 0]
    ...     blk[53][i].number_of_complex_zeros = 3
    ...     blk[53][i].real_zero = [0.0, 0.0, 0.0]
    ...     blk[53][i].imaginary_zero = [0.0, 0.0, 0.0]
    ...     blk[53][i].real_zero_error = [0, 0, 0]
    ...     blk[53][i].imaginary_zero_error = [0, 0, 0]
    ...     blk[53][i].A0_normalization_factor = 1.0
    ...     blk[53][i].normalization_frequency = 3.0
    ...     # stage sequence number 1, seismometer gain
    ...     blk[58][i*mult].sensitivity_gain = 400.0
    ...     # stage sequence number 2, digitizer gain
    ...     blk[58][i*mult+1].sensitivity_gain = 1677850.0
    ...     # stage sequence number 0, overall sensitivity
    ...     blk[58][(i+1)*mult-1].sensitivity_gain = 671140000.0

.. note::

    FIR coefficients are not set in this example. In case you require correct FIR coefficients, either clone from an existing dataless file with the same seismometer type or set the corresponding blockettes with the correct values.

At the end we can write the adapted DatalessSEED volume to a new file:

    >>> p.write_seed("dataless.seed.BW_RMOA")


.. _`examples webserver`: https://examples.obspy.org
