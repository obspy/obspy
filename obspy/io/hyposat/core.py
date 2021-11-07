# -*- coding: utf-8 -*-
"""
HYPOSAT phase file format write support.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import copy
import warnings


def _write_hyposat_phases(
        catalog, path_or_file_object, hyposat_rename_first_onsets=False,
        **kwargs):
    """
    Write a HYPOSAT phases file (called "hyposat-in" in HYPOSAT manual) from a
    :class:`~obspy.core.event.Catalog` object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.event.Catalog.write` method of an
        ObsPy :class:`~obspy.core.event.Catalog` object, call this instead.

    :type catalog: :class:`~obspy.core.stream.Catalog`
    :param catalog: The ObsPy Catalog object to write. The catalog should only
        contain one single Event object.
    :type filename: str or file
    :param filename: Filename to write or open file-like object.
    :type hyposat_rename_first_onsets: bool
    :param hyposat_rename_first_onsets: Whether to relabel first "P" and "S"
        picks, see :func:`_rename_phase_hints_as_first_onset`.
    """
    if len(catalog) > 1:
        msg = ("Writing NonLinLoc Phase file is only supported for Catalogs "
               "with a single Event in it (use a for loop over the catalog "
               "and provide an output file name for each event).")
        raise ValueError(msg)

    event = catalog[0]
    picks = event.picks

    if hyposat_rename_first_onsets:
        picks = _rename_phase_hints_as_first_onset(picks)

    event_info = str(event.resource_id)
    try:
        event_info += ' ' + str(event).split("\n", 1)[0].split("\t")[1]
    except Exception:
        pass
    # limit to 80 characters as per HYPOSAT manual
    event_info = event_info[:80]

    # From HYPOSAT manual:
    # S-type onsets must always be listed after the corresponding P-type
    # onsets - if not, the travel- time difference between these two onset
    # types (S-P) cannot be used for calculating an initial solution for the
    # source time with the Wadati approach and/or the distance from the
    # corresponding station.
    # Simply sorting picks by time should handle this.
    lines = [_pick_to_hyposat_phase_line(pick)
             for pick in sorted(picks, key=lambda p: p.time)]
    newline = '\n'
    out = newline.join([event_info] + lines)
    out = out.encode('ASCII')

    if not hasattr(path_or_file_object, 'write'):
        f = open(path_or_file_object, 'wb')
    else:
        f = path_or_file_object
    try:
        f.write(out)
    finally:
        if not hasattr(path_or_file_object, 'write'):
            f.close()


def _pick_to_hyposat_phase_line(pick):
    """
    Return HYPOSAT phase line representation of pick

    :type pick: :class:`~obspy.core.event.origin.Pick`
    :rtype: str
    """
    # a5,1x       station name
    # a8,1x       phase name
    # i4,1x       year
    # 4(i2,1x)    month, day, hour, minute
    # f6.3,1x     second
    # f5.3,1x     standard deviation of the onset time
    # f6.2,1x     backazimuth
    # 3(f5.2,1x)  standard deviation of the backazimuth observation, either
    #             ray parameter [s/deg] or apparent velocity [km/s], standard
    #             deviation of the slowness observation in [s/deg] or [km/s]
    # a7,1x       seven character long combination of controlling flags
    # f6.3,1x     the period of the observed onset
    # f12.2,1x    amplitude of the signal in [nm]
    # f7.2,1x     signal-to-noise ratio (SNR)
    # a8,1x       eight-character long arrival id
    # f5.2        second standard deviation for the onset time reading

    if pick.waveform_id.station_code is None:
        # QuakeML has station code mandatory, but we do not currently
        msg = f'Using pick without station code: {pick.resource_id}'
        warnings.warn(msg)
        station = ' ' * 5
    else:
        station = pick.waveform_id.station_code
        if len(station) > 5:
            msg = (f"Station code exceeds five characters, getting truncated: "
                   f"'{station}'")
            warnings.warn(msg)
        station = station[:5]
    if pick.phase_hint is None:
        msg = f'Using pick without phase hint: {pick.resource_id}'
        warnings.warn(msg)
        phase = ' ' * 8
    else:
        phase = pick.phase_hint
        if len(phase) > 8:
            msg = (f"Phase hint exceeds eight characters, getting truncated: "
                   f"'{phase}'")
            warnings.warn(msg)
        phase = phase[:8]
    time = pick.time
    std_time = None
    std_time2 = None
    if pick.time_errors:
        std_time = pick.time_errors.uncertainty
        std_time2 = pick.time_errors.uncertainty
        if pick.time_errors.lower_uncertainty is not None:
            std_time = pick.time_errors.lower_uncertainty
        if pick.time_errors.upper_uncertainty is not None:
            std_time2 = pick.time_errors.upper_uncertainty
    if std_time is None:
        std_time = ' ' * 5
    else:
        std_time = f'{std_time:5.3f}'
    if std_time2 is None:
        std_time2 = ' ' * 5
    else:
        std_time2 = f'{std_time2:5.3f}'
    if pick.backazimuth is None:
        baz = ' ' * 6
    else:
        baz = f'{pick.backazimuth:6.2f}'
    # XXX should handle lower/upper if general uncertainty one is not set
    if pick.backazimuth_errors \
            and pick.backazimuth_errors.uncertainty is not None:
        std_baz = f'{pick.backazimuth_errors.uncertainty:5.2f}'
    else:
        std_baz = ' ' * 5
    if pick.horizontal_slowness is None:
        slowness = ' ' * 5
    else:
        slowness = f'{pick.horizontal_slowness:5.2f}'
    # XXX should handle lower/upper if general uncertainty one is not set
    if pick.horizontal_slowness_errors \
            and pick.horizontal_slowness_errors.uncertainty is not None:
        std_slowness = f'{pick.horizontal_slowness_errors.uncertainty:5.2f}'
    else:
        std_slowness = ' ' * 5
    # control flags are not determined from the pick, but rather would be set
    # by a user manually. so either make it controlable via args/kwargs or just
    # keep it empty which then uses the default when used in HYPOSAT:
    # "If keeping positions 1-7 blank, the flag combination TASDRM_ will be
    # used as default value."
    control_flags = ' ' * 7
    # no period info in Pick
    onset_period = ' ' * 6
    # XXX amplitude would have to be looked up from a corresponding Amplitude
    # XXX object but questionable if we could make sure that its nanometers, so
    # XXX skip
    amplitude_nm = ' ' * 12
    snr = ' ' * 7
    # pick resource IDs are usually way longer than 8 characters, so skip it
    arrival_id = ' ' * 8

    line = (f'{station:<5s} {phase:<8s} {time.year:4d} {time.month:02d} '
            f'{time.day:02d} {time.hour:02d} {time.minute:02d} '
            f'{time.second + time.microsecond / 1e6:6.3f} {std_time} {baz} '
            f'{std_baz} {slowness} {std_slowness} {control_flags:7s} '
            f'{onset_period} {amplitude_nm} {snr} {arrival_id} {std_time2}')
    return line


def _rename_phase_hints_as_first_onset(picks):
    """
    Rename "P" and "S" picks to "P1" and "S1".

    Rename picks with phase hints "P" and "S" to "P1" and "S1", respectively,
    if they are the only picks with that `phase_hint` at a given station.
    Returns a new list of picks with deep copies of original picks.

    From HYPOSAT manual::

        If the type of the P or S onset you have is unknown, you can choose the
        names P1 or S1 to tell the program that you know it is the first P-type
        or the first S-type onset at this station. Then the program chooses the
        right phase name depending on the epicentral distance of the
        observation and the travel-time table.

    :type picks: list
    :param picks: List of input picks
    :rtype: list of :class:`~obspy.core.event.origin.Pick`
    """
    picks = [copy.deepcopy(pick) for pick in picks]
    picks_per_station = {}
    for pick in picks:
        net_sta_loc = pick.waveform_id.get_seed_string().rsplit('.', 1)[0]
        picks_per_station.setdefault(net_sta_loc, []).append(pick)
    renamed = []
    # group picks by common network.station.location part of SEED string
    for net_sta_loc, picks in picks_per_station.items():
        p_picks = []
        s_picks = []
        others = []
        for pick in picks:
            if pick.phase_hint == 'P':
                p_picks.append(pick)
            elif pick.phase_hint == 'S':
                s_picks.append(pick)
            else:
                others.append(pick)
        # check if only one "P" or "S" and rename if thats the case
        if len(p_picks) == 1:
            p_picks[0].phase_hint = 'P1'
        if len(s_picks) == 1:
            s_picks[0].phase_hint = 'S1'
        renamed.extend(p_picks)
        renamed.extend(s_picks)
        renamed.extend(others)
    return renamed


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
