# -*- coding: utf-8 -*-
"""
CNV file format support for ObsPy

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

from bisect import bisect_right
import warnings


def _write_cnv(catalog, filename, phase_mapping=None, ifx_list=None,
               weight_mapping=None, default_weight=0):
    """
    Write a :class:`~obspy.core.event.Catalog` object to CNV event summary
    format (used as event/pick input by VELEST program).

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.event.Catalog.write` method of an
        ObsPy :class:`~obspy.core.event.Catalog` object, call this instead.

    :type catalog: :class:`~obspy.core.event.Catalog`
    :param catalog: Input catalog for CNV output..
    :type filename: str or file
    :param filename: Filename to write or open file-like object.
    :type phase_mapping: dict
    :param phase_mapping: Mapping of phase hints to "P" or "S". CNV format only
        uses a single letter phase code (either "P" or "S"). If not specified
        the following default mapping is used: 'p', 'P', 'Pg', 'Pn', 'Pm' will
        be mapped to "P" and 's', 'S', 'Sg', 'Sn', 'Sm' will be mapped to "S".
    :type ifx_list: list of
        :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param ifx_list: List of events for which the 'IFX' flag should be set
        (used in VELEST to fix the y coordinate of the hypocenter).
    :type weight_mapping: list of float
    :param weight_mapping: Mapping of pick uncertainties to integer weights.
        (Sorted) list of floats of boundary uncertainties. If uncertainty of
        pick is lower than the first entry of the list than a weight of 0 is
        assigned. If it is larger than the first entry, but smaller than the
        second entry a weight of 1 is assigned, and so on. The list of
        uncertainty boundaries should not contain more than 9 entries because
        the integer weight is restricted to a single digit. If not specified
        all picks will be output with weight `default_weight`.
    :type default_weight: int
    :param default_weight: Default weight to use when pick has no timing
        uncertainty and thus can not be mapped using `weight_mapping`
        parameter. Default weight should not be larger than 9, as the weight is
        represented as a single digit.
    """
    # Check phase mapping or use default one
    if phase_mapping is None:
        phase_mapping = {'p': "P", 'P': "P", 'Pg': "P", 'Pn': "P", 'Pm': "P",
                         's': "S", 'S': "S", 'Sg': "S", 'Sn': "S", 'Sm': "S"}
    else:
        values = set(phase_mapping.values())
        values.update(("P", "S"))
        if values != set(("P", "S")):
            msg = ("Values of phase mapping should only be 'P' or 'S'")
            raise ValueError(msg)
    if ifx_list is None:
        ifx_list = []
    if weight_mapping is None:
        weight_mapping = []
    else:
        if list(weight_mapping) != sorted(weight_mapping):
            msg = ("List of floats in weight mapping must be sorted in "
                   "ascending order.")
            raise ValueError(msg)

    out = []
    for event in catalog:
        o = event.preferred_origin() or event.origins[0]
        m = event.preferred_magnitude() or event.magnitudes[0]

        out_ = "%s %5.2f %7.4f%1s %8.4f%1s%7.2f%7.2f%2i\n"
        cns = o.latitude >= 0 and "N" or "S"
        cew = o.longitude >= 0 and "E" or "W"
        if event.resource_id in ifx_list:
            ifx = 1
        else:
            ifx = 0
        out_ = out_ % (o.time.strftime("%y%m%d %H%M"),
                       o.time.second + o.time.microsecond / 1e6,
                       abs(o.latitude), cns, abs(o.longitude), cew,
                       o.depth / 1e3, m.mag, ifx)
        # assemble phase info
        picks = []
        for p in event.picks:
            # map uncertainty to integer weight
            if p.time_errors.upper_uncertainty is not None and \
                    p.time_errors.lower_uncertainty is not None:
                uncertainty = p.time_errors.upper_uncertainty + \
                    p.time_errors.lower_uncertainty
            else:
                uncertainty = p.time_errors.uncertainty
            if uncertainty is None:
                msg = ("No pick time uncertainty, pick will be mapped to "
                       "default integer weight (%s).") % default_weight
                warnings.warn(msg)
                weight = default_weight
            else:
                weight = bisect_right(weight_mapping, uncertainty)
            if weight > 9:
                msg = ("Integer weight for pick is greater than 9. "
                       "This is not compatible with the single-character "
                       "field for pick weight in CNV format."
                       "Using 9 as integer weight.")
                warnings.warn(msg)
                weight = 9
            # map phase hint
            phase = phase_mapping.get(p.phase_hint, None)
            if phase is None:
                msg = "Skipping pick (%s) with unmapped phase hint: %s"
                msg = msg % (p.waveform_id.get_seed_string(), p.phase_hint)
                warnings.warn(msg)
                continue
            station = p.waveform_id.station_code
            if len(station) > 4:
                msg = ("Station code with more than 4 characters detected. "
                       "Only the first 4 characters will be used in output.")
                warnings.warn(msg)
                station = station[:4]
            dt = "%6.2f" % (p.time - o.time)
            if len(dt) != 6:
                msg = ("Problem with pick (%s): Calculated travel time '%s' "
                       "does not fit in the '%%6.2f' fixed format field. "
                       "Skipping this pick.")
                msg = msg % (p.waveform_id.get_seed_string(), dt)
                warnings.warn(msg)
                continue
            picks.append("".join([station.ljust(4), phase, str(weight), dt]))
        while len(picks) > 6:
            next_picks, picks = picks[:6], picks[6:]
            out_ += "".join(next_picks) + "\n"
        if picks:
            out_ += "".join(picks) + "\n"
        out.append(out_)

    if out:
        out = "\n".join(out + [""])
    else:
        msg = "No event/pick information, writing empty CNV file."
        warnings.warn(msg)

    # Open filehandler or use an existing file like object.
    if not hasattr(filename, "write"):
        file_opened = True
        fh = open(filename, "wb")
    else:
        file_opened = False
        fh = filename

    fh.write(out.encode())

    # Close if a file has been opened by this function.
    if file_opened is True:
        fh.close()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
