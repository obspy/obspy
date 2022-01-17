"""
Receiver Gather (version 1.6-1) bindings to ObsPy core module.
"""
from collections import namedtuple

import numpy as np

from obspy.core import Stream, Trace, Stats, UTCDateTime
from obspy.io.rg16.util import _read, _open_file, _quick_merge


HeaderCount = namedtuple('HeaderCount', 'channel_set extended external')


@_open_file
def _read_rg16(filename, headonly=False, starttime=None, endtime=None,
               merge=False, contacts_north=False, details=False, **kwargs):
    """
    Read Fairfield Nodal's Receiver Gather File Format version 1.6-1.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param filename: path to the rg16 file or a file object.
    :type filename: str or file-like object
    :param headonly: If True don't read data, only main information
        contained in the headers of the trace block is read.
    :type headonly: optional, bool
    :param starttime: If not None dont read traces that start before starttime.
    :type starttime: optional, :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param endtime: If not None dont read traces that start after endtime.
    :type endtime: optional, :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param merge: If True merge contiguous data blocks as they are found. For
        continuous data files having 100,000+ traces this will create
        more manageable streams.
    :type merge: bool
    :param contacts_north: If this parameter is set to True, it will map the
        components to Z (1C, 3C), N (3C), and E (3C) as well as correct
        the polarity for the vertical component.
    :type contacts_north: bool
    :param details: If True, all the information contained in the headers
        is read).
    :type details: bool
    :return: An ObsPy :class:`~obspy.core.stream.Stream` object.
        Frequencies are expressed in hertz and time is expressed in second
        (except for date).
    """
    starttime = starttime or UTCDateTime(1970, 1, 1)
    endtime = endtime or UTCDateTime()
    # get the number of headers/records, position of trace data
    # and record length.
    header_count = _cmp_nbr_headers(filename)
    record_count = _cmp_nbr_records(filename)
    trace_block_start = 32 * (2 + sum(header_count))
    record_length = _cmp_record_length(filename)
    # create trace data
    traces = []
    for i in range(0, record_count):
        nbr_bytes_trace_block = _cmp_jump(filename, trace_block_start)
        trace_starttime = _read(filename, trace_block_start + 20 + 2 * 32, 8,
                                'binary') / 1e6
        trace_endtime = trace_starttime + record_length
        con1 = starttime.timestamp > trace_endtime
        con2 = endtime.timestamp < trace_starttime
        # determine if this block is excluded based in starttime/endtime params
        if con1 or con2:
            # the block has to be excluded, increment trace_block_start
            #  and continue
            trace_block_start += nbr_bytes_trace_block
            continue
        trace = _make_trace(filename, trace_block_start, headonly,
                            contacts_north, details)
        traces.append(trace)
        trace_block_start += nbr_bytes_trace_block
    if merge:
        traces = _quick_merge(traces)
    return Stream(traces=traces)


def _cmp_nbr_headers(fi):
    """
    Return a tuple containing the number of channel set headers,
    the number of extended headers and the number of external headers
    in the file.
    """
    header_count = HeaderCount(
        channel_set=_read(fi, 28, 1, 'bcd'),
        extended=_read(fi, 37, 2, 'binary'),
        external=_read(fi, 39, 3, 'binary'),
    )
    return header_count


def _cmp_nbr_records(fi):
    """
    Return the number of records in the file (ie number of time slices
    multiplied by the number of components).
    """
    initial_header = _read_initial_headers(fi)
    channel_sets_descriptor = initial_header['channel_sets_descriptor']
    channels_number = set()

    for _, val in channel_sets_descriptor.items():
        chan_num = val['RU_channel_number'] or val['channel_set_number']
        channels_number.add(chan_num)

    nbr_component = len(channels_number)
    extended_header_2 = initial_header['extended_headers']['2']
    nbr_time_slices = extended_header_2['nbr_time_slices']
    nbr_records = nbr_time_slices * nbr_component
    return nbr_records


def _cmp_record_length(fi):
    """
    Return the record length.
    """
    base_scan_interval = _read(fi, 22, 1, 'binary')
    sampling_rate = int(1000 / (base_scan_interval / 16))
    gen_head_2 = _read_initial_headers(fi)['general_header_2']
    record_length = gen_head_2['extended_record_length'] - 1 / sampling_rate
    return record_length


def _cmp_jump(fi, trace_block_start):
    """
    Return the number of bytes in a trace block.
    """
    nbr_trace_extension_block = _read(fi, trace_block_start + 9, 1, 'binary')
    nbr_bytes_header_trace = 20 + 32 * nbr_trace_extension_block
    nbr_sample_trace = _read(fi, trace_block_start + 27, 3, 'binary')
    nbr_bytes_trace_data = nbr_sample_trace * 4
    nbr_bytes_trace_block = nbr_bytes_trace_data + nbr_bytes_header_trace
    return nbr_bytes_trace_block


def _make_trace(fi, trace_block_start, headonly, standard_orientation,
                details):
    """
    Make obspy trace from a trace block (header + trace).
    """
    stats = _make_stats(fi, trace_block_start, standard_orientation, details)
    if headonly:
        data = np.array([])
    else:  # read trace
        nbr_trace_extension_block = _read(fi, trace_block_start + 9,
                                          1, 'binary')
        trace_start = trace_block_start + 20 + nbr_trace_extension_block * 32
        nbr_sample_trace = _read(fi, trace_block_start + 27, 3, 'binary')
        nbr_bytes_trace = 4 * nbr_sample_trace
        data = _read(fi, trace_start, nbr_bytes_trace, 'IEEE')
        if stats.channel[-1] == 'Z':
            data = -data
            data = data.astype('>f4')
    return Trace(data=data, header=stats)


def _make_stats(fi, tr_block_start, standard_orientation, details):
    """
    Make Stats object from information contained in the header of the trace.
    """
    base_scan_interval = _read(fi, 22, 1, 'binary')
    sampling_rate = int(1000 / (base_scan_interval / 16))
    # map sampling rate to band code according to seed standard
    band_map = {2000: 'G', 1000: 'G', 500: 'D', 250: 'D'}
    # geophone instrument code
    instrument_code = 'P'
    # mapping for "standard_orientation"
    standard_component_map = {'2': 'Z', '3': 'N', '4': 'E'}
    component = str(_read(fi, tr_block_start + 40, 1, 'binary'))
    if standard_orientation:
        component = standard_component_map[component]
    chan = band_map[sampling_rate] + instrument_code + component
    npts = _read(fi, tr_block_start + 27, 3, 'binary')
    start_time = _read(fi, tr_block_start + 20 + 2 * 32, 8, 'binary') / 1e6
    end_time = start_time + (npts - 1) * (1 / sampling_rate)
    network = _read(fi, tr_block_start + 20, 3, 'binary')
    station = _read(fi, tr_block_start + 23, 3, 'binary')
    location = _read(fi, tr_block_start + 26, 1, 'binary')
    statsdict = dict(starttime=UTCDateTime(start_time),
                     endtime=UTCDateTime(end_time),
                     sampling_rate=sampling_rate,
                     npts=npts,
                     network=str(network),
                     station=str(station),
                     location=str(location),
                     channel=chan)
    if details:
        statsdict['rg16'] = {}
        statsdict['rg16']['initial_headers'] = {}
        stats_initial_headers = statsdict['rg16']['initial_headers']
        stats_initial_headers.update(_read_initial_headers(fi))
        statsdict['rg16']['trace_headers'] = {}
        stats_tr_headers = statsdict['rg16']['trace_headers']
        stats_tr_headers.update(_read_trace_header(fi, tr_block_start))
        nbr_tr_header_block = _read(fi, tr_block_start + 9,
                                    1, 'binary')
        if nbr_tr_header_block > 0:
            stats_tr_headers.update(
                _read_trace_headers(fi, tr_block_start, nbr_tr_header_block))
    return Stats(statsdict)


def _read_trace_headers(fi, trace_block_start, nbr_trace_header):
    """
    Read headers in the trace block.
    """
    trace_headers = {}
    dict_func = {'1': _read_trace_header_1, '2': _read_trace_header_2,
                 '3': _read_trace_header_3, '4': _read_trace_header_4,
                 '5': _read_trace_header_5, '6': _read_trace_header_6,
                 '7': _read_trace_header_7, '8': _read_trace_header_8,
                 '9': _read_trace_header_9, '10': _read_trace_header_10}
    for i in range(1, nbr_trace_header + 1):
        trace_headers.update(dict_func[str(i)](fi, trace_block_start))
    return trace_headers


def _read_trace_header(fi, trace_block_start):
    """
    Read the 20 bytes trace header (first header in the trace block).
    """
    trace_number = _read(fi, trace_block_start + 4, 2, 'bcd')
    trace_edit_code = _read(fi, trace_block_start + 11, 1, 'binary')
    return {'trace_number': trace_number, 'trace_edit_code': trace_edit_code}


def _read_trace_header_1(fi, trace_block_start):
    """
    Read trace header 1
    """
    pos = trace_block_start + 20

    dict_header_1 = dict(
        extended_receiver_line_nbr=_read(fi, pos + 10, 5, 'binary'),
        extended_receiver_point_nbr=_read(fi, pos + 15, 5, 'binary'),
        sensor_type=_read(fi, pos + 20, 1, 'binary'),
        trace_count_file=_read(fi, pos + 21, 4, 'binary'),
    )
    return dict_header_1


def _read_trace_header_2(fi, trace_block_start):
    """
    Read trace header 2
    """
    pos = trace_block_start + 20 + 32

    leg_source_info = {'0': 'undefined', '1': 'preplan', '2': 'as shot',
                       '3': 'post processed'}
    source_key = str(_read(fi, pos + 29, 1, 'binary'))

    leg_energy_source = {'0': 'undefined', '1': 'vibroseis', '2': 'dynamite',
                         '3': 'air gun'}
    energy_source_key = str(_read(fi, pos + 30, 1, 'binary'))

    dict_header_2 = dict(
        shot_line_nbr=_read(fi, pos, 4, 'binary'),
        shot_point=_read(fi, pos + 4, 4, 'binary'),
        shot_point_index=_read(fi, pos + 8, 1, 'binary'),
        shot_point_pre_plan_x=_read(fi, pos + 9, 4, 'binary') / 10,
        shot_point_pre_plan_y=_read(fi, pos + 13, 4, 'binary') / 10,
        shot_point_final_x=_read(fi, pos + 17, 4, 'binary') / 10,
        shot_point_final_y=_read(fi, pos + 21, 4, 'binary') / 10,
        shot_point_final_depth=_read(fi, pos + 25, 4, 'binary') / 10,
        source_of_final_shot_info=leg_source_info[source_key],
        energy_source_type=leg_energy_source[energy_source_key],
    )
    return dict_header_2


def _read_trace_header_3(fi, trace_block_start):
    """
    Read trace header 3
    """
    pos = trace_block_start + 20 + 32 * 2

    dict_header_3 = dict(
        epoch_time=UTCDateTime(_read(fi, pos, 8, 'binary') / 1e6),
        # shot skew time in second
        shot_skew_time=_read(fi, pos + 8, 8, 'binary') / 1e6,
        # time shift clock correction in second
        time_shift_clock_correction=_read(fi, pos + 16, 8, 'binary') / 1e9,
        # remaining clock correction in second
        remaining_clock_correction=_read(fi, pos + 24, 8, 'binary') / 1e9,
    )
    return dict_header_3


def _read_trace_header_4(fi, trace_block_start):
    """
    Read trace header 4
    """
    pos = trace_block_start + 20 + 32 * 3

    leg_trace_clipped = {'0': 'not clipped', '1': 'digital clip detected',
                         '2': 'analog clip detected'}
    clipped_code = str(_read(fi, pos + 9, 1, 'binary'))

    leg_record_type = {'2': 'test data record',
                       '8': 'normal seismic data record'}
    record_type_code = str(_read(fi, pos + 10, 1, 'binary'))

    leg_shot_flag = {'0': 'normal', '1': 'bad-operator specified',
                     '2': 'bad-failed to QC test'}
    shot_code = str(_read(fi, pos + 11, 1, 'binary'))

    dict_header_4 = dict(
        # pre shot guard band in second
        pre_shot_guard_band=_read(fi, pos, 4, 'binary') / 1e3,
        # post shot guard band in second
        post_shot_guard_band=_read(fi, pos + 4, 4, 'binary') / 1e3,
        # preamp gain in dB
        preamp_gain=_read(fi, pos + 8, 1, 'binary'),
        trace_clipped_flag=leg_trace_clipped[clipped_code],
        record_type_code=leg_record_type[record_type_code],
        shot_status_flag=leg_shot_flag[shot_code],
        external_shot_id=_read(fi, pos + 12, 4, 'binary'),
        post_processed_first_break_pick_time=_read(fi, pos + 24, 4, 'IEEE'),
        post_processed_rms_noise=_read(fi, pos + 28, 4, 'IEEE'),
    )
    return dict_header_4


def _read_trace_header_5(fi, trace_block_start):
    """
    Read trace header 5
    """
    pos = trace_block_start + 20 + 32 * 4

    leg_source_receiver_info = {
        '1': 'preplan',
        '2': 'as laid (no navigation sensor)',
        '3': 'as laid (HiPAP only)',
        '4': 'as laid (HiPAP and INS)',
        '5': 'as laid (HiPAP and DVL)',
        '6': 'as laid (HiPAP, DVL and INS)',
        '7': 'post processed (HiPAP only)',
        '8': 'post processed (HiPAP and INS)',
        '9': 'post processed (HiPAP and DVL)',
        '10': 'post processed (HiPAP, DVL ans INS)',
        '11': 'first break analysis',
    }
    source_key = str(_read(fi, pos + 29, 1, 'binary'))

    dict_header_5 = dict(
        receiver_point_pre_plan_x=_read(fi, pos + 9, 4, 'binary') / 10,
        receiver_point_pre_plan_y=_read(fi, pos + 13, 4, 'binary') / 10,
        receiver_point_final_x=_read(fi, pos + 17, 4, 'binary') / 10,
        receiver_point_final_y=_read(fi, pos + 21, 4, 'binary') / 10,
        receiver_point_final_depth=_read(fi, pos + 25, 4, 'binary') / 10,
        source_of_final_receiver_info=leg_source_receiver_info[source_key],
    )
    return dict_header_5


def _read_trace_header_6(fi, trace_block_start):
    """
    Read trace header 6
    """
    pos = trace_block_start + 20 + 32 * 5

    dict_header_6 = dict(
        tilt_matrix_h1x=_read(fi, pos, 4, 'IEEE'),
        tilt_matrix_h2x=_read(fi, pos + 4, 4, 'IEEE'),
        tilt_matrix_vx=_read(fi, pos + 8, 4, 'IEEE'),
        tilt_matrix_h1y=_read(fi, pos + 12, 4, 'IEEE'),
        tilt_matrix_h2y=_read(fi, pos + 16, 4, 'IEEE'),
        tilt_matrix_vy=_read(fi, pos + 20, 4, 'IEEE'),
        tilt_matrix_h1z=_read(fi, pos + 24, 4, 'IEEE'),
        tilt_matrix_h2z=_read(fi, pos + 28, 4, 'IEEE'),
    )
    return dict_header_6


def _read_trace_header_7(fi, trace_block_start):
    """
    Read trace header 7
    """
    pos = trace_block_start + 20 + 32 * 6

    dict_header_7 = dict(
        tilt_matrix_vz=_read(fi, pos, 4, 'IEEE'),
        azimuth_degree=_read(fi, pos + 4, 4, 'IEEE'),
        pitch_degree=_read(fi, pos + 8, 4, 'IEEE'),
        roll_degree=_read(fi, pos + 12, 4, 'IEEE'),
        remote_unit_temp=_read(fi, pos + 16, 4, 'IEEE'),
        remote_unit_humidity=_read(fi, pos + 20, 4, 'IEEE'),
        orientation_matrix_version_nbr=_read(fi, pos + 24, 4, 'binary'),
        gimbal_corrections=_read(fi, pos + 28, 1, 'binary'))
    return dict_header_7


def _read_trace_header_8(fi, trace_block_start):
    """
    Read trace header 8
    """
    pos = trace_block_start + 20 + 32 * 7

    leg_preamp_path = {
        '0': 'external input selected',
        '1': 'simulated data selected',
        '2': 'pre-amp input shorted to ground',
        '3': 'test oscillator with sensors',
        '4': 'test oscillator without sensors',
        '5': 'common mode test oscillator with sensors',
        '6': 'common mode test oscillator without sensors',
        '7': 'test oscillator on positive sensors with neg sensor grounded',
        '8': 'test oscillator on negative sensors with pos sensor grounded',
        '9': 'test oscillator on positive PA input with neg PA input ground',
        '10': 'test oscillator on negative PA input with pos PA input ground',
        '11': 'test oscillator on positive PA input with neg\
                              PA input ground, no sensors',
        '12': 'test oscillator on negative PA input with pos\
                              PA input ground, no sensors'}
    preamp_path_code = str(_read(fi, pos + 24, 4, 'binary'))

    leg_test_oscillator = {'0': 'test oscillator path open',
                           '1': 'test signal selected',
                           '2': 'DC reference selected',
                           '3': 'test oscillator path grounded',
                           '4': 'DC reference toggle selected'}
    oscillator_code = str(_read(fi, pos + 28, 4, 'binary'))

    dict_header_8 = dict(
        fairfield_test_analysis_code=_read(fi, pos, 4, 'binary'),
        first_test_oscillator_attenuation=_read(fi, pos + 4, 4, 'binary'),
        second_test_oscillator_attenuation=_read(fi, pos + 8, 4, 'binary'),
        # start delay in second
        start_delay=_read(fi, pos + 12, 4, 'binary') / 1e6,
        dc_filter_flag=_read(fi, pos + 16, 4, 'binary'),
        dc_filter_frequency=_read(fi, pos + 20, 4, 'IEEE'),
        preamp_path=leg_preamp_path[preamp_path_code],
        test_oscillator_signal_type=leg_test_oscillator[oscillator_code],
    )
    return dict_header_8


def _read_trace_header_9(fi, trace_block_start):
    """
    Read trace header 9
    """
    pos = trace_block_start + 20 + 32 * 8

    leg_signal_type = {'0': 'pattern is address ramp',
                       '1': 'pattern is RU address ramp',
                       '2': 'pattern is built from provided values',
                       '3': 'pattern is random numbers',
                       '4': 'pattern is a walking 1s',
                       '5': 'pattern is a walking 0s',
                       '6': 'test signal is a specified DC value',
                       '7': 'test signal is a pulse train with\
                             specified duty cycle',
                       '8': 'test signal is a sine wave',
                       '9': 'test signal is a dual tone sine',
                       '10': 'test signal is an impulse',
                       '11': 'test signal is a step function'}
    type_code = str(_read(fi, pos, 4, 'binary'))

    # test signal generator frequency 1 in hertz
    test_signal_freq_1 = _read(fi, pos + 4, 4, 'binary') / 1e3
    # test signal generator frequency 2 in hertz
    test_signal_freq_2 = _read(fi, pos + 8, 4, 'binary') / 1e3
    # test signal generator amplitude 1 in dB down from full scale -120 to 120
    test_signal_amp_1 = _read(fi, pos + 12, 4, 'binary')
    # test signal generator amplitude 2 in dB down from full scale -120 to 120
    test_signal_amp_2 = _read(fi, pos + 16, 4, 'binary')
    # test signal generator duty cycle in percentage
    duty_cycle = _read(fi, pos + 20, 4, 'IEEE')
    # test signal generator active duration in second
    active_duration = _read(fi, pos + 24, 4, 'binary') / 1e6
    # test signal generator activation time in second
    activation_time = _read(fi, pos + 28, 4, 'binary') / 1e6

    dict_header_9 = dict(
        test_signal_generator_signal_type=leg_signal_type[type_code],
        test_signal_generator_frequency_1=test_signal_freq_1,
        test_signal_generator_frequency_2=test_signal_freq_2,
        test_signal_generator_amplitude_1=test_signal_amp_1,
        test_signal_generator_amplitude_2=test_signal_amp_2,
        test_signal_generator_duty_cycle_percentage=duty_cycle,
        test_signal_generator_active_duration=active_duration,
        test_signal_generator_activation_time=activation_time,
    )
    return dict_header_9


def _read_trace_header_10(fi, trace_block_start):
    """
    Read trace header 10
    """
    pos = trace_block_start + 20 + 32 * 9

    dict_header_10 = dict(
        test_signal_generator_idle_level=_read(fi, pos, 4, 'binary'),
        test_signal_generator_active_level=_read(fi, pos + 4, 4, 'binary'),
        test_signal_generator_pattern_1=_read(fi, pos + 8, 4, 'binary'),
        test_signal_generator_pattern_2=_read(fi, pos + 12, 4, 'binary'),
    )
    return dict_header_10


@_open_file
def _is_rg16(filename, **kwargs):
    """
    Determine if a file is a rg16 file.

    :param filename: a path to a file or a file object
    :type filename: str or file-like object
    :rtype: bool
    :return: True if the file object is a rg16 file.
    """
    try:
        sample_format = _read(filename, 2, 2, 'bcd')
        sample_format = _read(filename, 2, 2, 'bcd')
        manufacturer_code = _read(filename, 16, 1, 'bcd')
        version = _read(filename, 42, 2, 'binary')
    except ValueError:  # if file too small
        return False
    con1 = version == 262 and sample_format == 8058
    return con1 and manufacturer_code == 20


@_open_file
def _read_initial_headers(filename):
    """
    Extract all the information contained in the headers located before data,
    at the beginning of the rg16 file object.

    :param filename: a path to a rg16 file or a rg16 file object.
    :type filename: str or file-like object
    :return: a dictionary containing all the information of the initial headers

    Frequencies are expressed in hertz and time is expressed in second (except
    for the date).
    """
    headers_content = dict(
        general_header_1=_read_general_header_1(filename),
        general_header_2=_read_general_header_2(filename),
        channel_sets_descriptor=_read_channel_sets(filename),
        extended_headers=_read_extended_headers(filename),
    )
    return headers_content


def _read_general_header_1(fi):
    """
    Extract information contained in the general header block 1
    """
    gen_head_1 = dict(
        file_number=_read(fi, 0, 2, 'bcd'),
        sample_format_code=_read(fi, 2, 2, 'bcd'),
        general_constant=_read(fi, 4, 6, 'bcd'),
        time_slice_year=_read(fi, 10, 1, 'bcd'),
        nbr_add_general_header=_read(fi, 11, 0.5, 'bcd'),
        julian_day=_read(fi, 11, 1.5, 'bcd', False),
        time_slice=_read(fi, 13, 3, 'bcd'),
        manufacturer_code=_read(fi, 16, 1, 'bcd'),
        manufacturer_serial_number=_read(fi, 17, 2, 'bcd'),
        base_scan_interval=_read(fi, 22, 1, 'binary'),
        polarity_code=_read(fi, 23, 0.5, 'binary'),
        record_type=_read(fi, 25, 0.5, 'binary'),
        scan_type_per_record=_read(fi, 27, 1, 'bcd'),
        nbr_channel_set=_read(fi, 28, 1, 'bcd'),
        nbr_skew_block=_read(fi, 29, 1, 'bcd'),
    )
    return gen_head_1


def _read_general_header_2(fi):
    """
    Extract information contained in the general header block 2
    """
    gen_head_2 = dict(
        extended_file_number=_read(fi, 32, 3, 'binary'),
        extended_channel_sets_per_scan_type=_read(fi, 35, 2, 'binary'),
        extended_header_blocks=_read(fi, 37, 2, 'binary'),
        external_header_blocks=_read(fi, 39, 3, 'binary'),
        version_number=_read(fi, 42, 2, 'binary'),
        # extended record length in second
        extended_record_length=_read(fi, 46, 3, 'binary') / 1e3,
        general_header_block_number=_read(fi, 50, 1, 'binary'),
    )
    return gen_head_2


def _read_channel_sets(fi):
    """
    Extract information of all channel set descriptor blocks.
    """
    channel_sets = {}
    nbr_channel_set = _read(fi, 28, 1, 'bcd')
    start_byte = 64
    for i in range(0, nbr_channel_set):
        channel_set_name = str(i + 1)
        channel_sets[channel_set_name] = _read_channel_set(fi, start_byte)
        start_byte += 32
    return channel_sets


def _read_channel_set(fi, start_byte):
    """
    Extract information contained in the ith channel set descriptor.
    """
    nbr_32_ext = _read(fi, start_byte + 28, 0.5, 'binary', False)
    # first read alias freq. This can be written as BCD or int32
    try:
        alias_filter_freq = _read(fi, start_byte + 12, 2, 'bcd')
    except ValueError:
        alias_filter_freq = _read(fi, start_byte + 12, 2, '>i2')

    channel_set = dict(
        scan_type_number=_read(fi, start_byte, 1, 'bcd'),
        channel_set_number=_read(fi, start_byte + 1, 1, 'bcd'),
        channel_set_start_time=_read(fi, start_byte + 2, 2, 'binary') * 2e-3,
        channel_set_end_time=_read(fi, start_byte + 4, 2, 'binary') * 2e-3,
        optionnal_MP_factor=_read(fi, start_byte + 6, 1, 'binary'),
        mp_factor_descaler_multiplier=_read(fi, start_byte + 7, 1, 'binary'),
        nbr_channels_in_channel_set=_read(fi, start_byte + 8, 2, 'bcd'),
        channel_type_code=_read(fi, start_byte + 10, 0.5, 'binary'),
        nbr_sub_scans=_read(fi, start_byte + 11, 0.5, 'bcd'),
        gain_control_type=_read(fi, start_byte + 11, 0.5, 'bcd', False),
        alias_filter_frequency=alias_filter_freq,
        alias_filter_slope=_read(fi, start_byte + 14, 2, 'bcd'),
        low_cut_filter_freq=_read(fi, start_byte + 16, 2, 'bcd'),
        low_cut_filter_slope=_read(fi, start_byte + 18, 2, 'bcd'),
        notch_filter_freq=_read(fi, start_byte + 20, 2, 'bcd') / 10,
        notch_2_filter_freq=_read(fi, start_byte + 22, 2, 'bcd') / 10,
        notch_3_filter_freq=_read(fi, start_byte + 24, 2, 'bcd') / 10,
        extended_channel_set_number=_read(fi, start_byte + 26, 2, 'binary'),
        extended_header_flag=_read(fi, start_byte + 28, 0.5, 'binary'),
        nbr_32_byte_trace_header_extension=nbr_32_ext,
        vertical_stack_size=_read(fi, start_byte + 29, 1, 'binary'),
        RU_channel_number=_read(fi, start_byte + 30, 1, 'binary'),
        array_forming=_read(fi, start_byte + 31, 1, 'binary'),
    )
    return channel_set


def _read_extended_headers(fi):
    """
    Extract information from the extended headers.
    """
    extended_headers = {}
    nbr_channel_set = _read(fi, 28, 1, 'bcd')
    start_byte = 32 + 32 + 32 * nbr_channel_set
    extended_headers['1'] = _read_extended_header_1(fi, start_byte)
    start_byte += 32
    extended_headers['2'] = _read_extended_header_2(fi, start_byte)
    start_byte += 32
    extended_headers['3'] = _read_extended_header_3(fi, start_byte)
    nbr_extended_headers = _read(fi, 37, 2, 'binary', True)
    if nbr_extended_headers > 3:
        coeffs = extended_headers['2']['number_decimation_filter_coefficient']
        nbr_coeff_remain = coeffs % 8
        for i in range(3, nbr_extended_headers):
            start_byte += 32
            extended_header_name = str(i + 1)
            if i == nbr_extended_headers - 1:
                header = _read_extended_header(fi, start_byte, i + 1,
                                               nbr_coeff_remain)
                extended_headers[extended_header_name] = header
            else:
                header = _read_extended_header(fi, start_byte, i + 1, 8)
                extended_headers[extended_header_name] = header
    return extended_headers


def _read_extended_header_1(fi, start_byte):
    """
    Extract information contained in the extended header block number 1.
    """
    deployment_time = _read(fi, start_byte + 8, 8, 'binary') / 1e6
    pick_up_time = _read(fi, start_byte + 16, 8, 'binary') / 1e6
    start_time_ru = _read(fi, start_byte + 24, 8, 'binary') / 1e6

    extended_header_1 = dict(
        id_ru=_read(fi, start_byte, 8, 'binary'),
        deployment_time=UTCDateTime(deployment_time),
        pick_up_time=UTCDateTime(pick_up_time),
        start_time_ru=UTCDateTime(start_time_ru),
    )
    return extended_header_1


def _read_extended_header_2(fi, start_byte):
    """
    Extract information contained in the extended header block number 2.
    """
    # code mappings to meaning
    leg_clock_stop = {'0': 'normal', '1': 'storage full', '2': 'power loss',
                      '3': 'reboot'}
    stop_code = str(_read(fi, start_byte + 12, 1, 'binary'))

    leg_freq_drift = {'0': 'not within specification',
                      '1': 'within specification'}
    drift_code = str(_read(fi, start_byte + 13, 1, 'binary'))

    leg_oscillator_type = {'0': 'control board', '1': 'atomic',
                           '2': 'ovenized', '3': 'double ovenized',
                           '4': 'disciplined'}
    oscillator_code = str(_read(fi, start_byte + 14, 1, 'binary'))

    leg_data_collection = {'0': 'normal', '1': 'continuous',
                           '2': 'shot sliced with guard band'}
    data_collection_code = str(_read(fi, start_byte + 15, 1, 'binary'))

    leg_data_decimation = {'0': 'not decimated', '1': 'decimated data'}
    decimation_code = str(_read(fi, start_byte + 28, 1, 'binary'))

    extended_header_2 = dict(
        acquisition_drift_window=_read(fi, start_byte, 4, 'IEEE') * 1e-6,
        clock_drift=_read(fi, start_byte + 4, 8, 'binary') * 1e-9,
        clock_stop_method=leg_clock_stop[stop_code],
        frequency_drift=leg_freq_drift[drift_code],
        oscillator_type=leg_oscillator_type[oscillator_code],
        data_collection_method=leg_data_collection[data_collection_code],
        nbr_time_slices=_read(fi, start_byte + 16, 4, 'binary'),
        nbr_files=_read(fi, start_byte + 20, 4, 'binary'),
        file_number=_read(fi, start_byte + 24, 4, 'binary'),
        data_decimation=leg_data_decimation[decimation_code],
        original_base_scan_interval=_read(fi, start_byte + 29, 1, 'binary'),
        nbr_decimation_filter_coef=_read(fi, start_byte + 30, 2, 'binary'),
    )
    return extended_header_2


def _read_extended_header_3(fi, start_byte):
    """
    Extract information contained in the extended header block number 3.
    """
    extended_header_3 = dict(
        receiver_line_number=_read(fi, start_byte, 4, 'binary'),
        receiver_point=_read(fi, start_byte + 4, 4, 'binary'),
        receiver_point_index=_read(fi, start_byte + 8, 1, 'binary'),
        first_shot_line=_read(fi, start_byte + 9, 4, 'binary'),
        first_shot_point=_read(fi, start_byte + 13, 4, 'binary'),
        first_shot_point_index=_read(fi, start_byte + 17, 1, 'binary'),
        last_shot_line=_read(fi, start_byte + 18, 4, 'binary'),
        last_shot_point=_read(fi, start_byte + 22, 4, 'binary'),
        last_shot_point_index=_read(fi, start_byte + 26, 1, 'binary'),
    )
    return extended_header_3


def _read_extended_header(fi, start_byte, block_number, nbr_coeff):
    """
    Extract information contained in the ith extended header block (i>3).
    """
    extended_header = {}
    for i in range(0, nbr_coeff):
        key = 'coeff_' + str(i + 1)
        extended_header[key] = _read(fi, start_byte, 4, 'IEEE')
        start_byte += 4
    return extended_header


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
