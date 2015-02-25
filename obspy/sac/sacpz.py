"""
Module for SAC poles and zero (SACPZ) file I/O.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy import UTCDateTime
from obspy.station.response import paz_to_sacpz_string


def write_SACPZ(inventory, file_or_file_object):
    """
    Writes an inventory object to a SACPZ file.

    .. note::
        The channel DIP in the SACPZ comment fields is specified like defined
        by SEED (positive down from horizontal).

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.station.inventory.Inventory.write` method of an
        ObsPy :class:`~obspy.station.inventory.Inventory` object, call this
        instead.

    :type inventory: :class:`~obspy.station.inventory.Inventory`
    :param inventory: The inventory instance to be written.
    :param file_or_file_object: The file or file-like object to be written to.
    """
    out = []
    now = UTCDateTime()
    for net in inventory:
        for sta in net:
            for cha in sta:
                resp = cha.response
                sens = resp.instrument_sensitivity
                paz = resp.get_paz()
                input_unit = sens.input_units.upper()
                if input_unit == "M":
                    pass
                elif input_unit in ["M/S", "M/SEC"]:
                    paz.zeros.append(0j)
                elif input_unit in ["M/S**2", "M/SEC**2"]:
                    paz.zeros.extend([0j, 0j])
                else:
                    msg = "Encountered unrecognized input units in response: "
                    raise NotImplementedError(msg + str(input_unit))
                out.append("* " + "*" * 50)
                out.append("* NETWORK     : %s" % net.code)
                out.append("* STATION     : %s" % sta.code)
                out.append("* LOCATION    : %s" % cha.location_code)
                out.append("* CHANNEL     : %s" % cha.code)
                out.append("* CREATED     : %s" % now)
                out.append("* START       : %s" % cha.start_date)
                out.append("* END         : %s" % cha.end_date)
                out.append("* DESCRIPTION : %s" % sta.site.name)
                out.append("* LATITUDE    : %s" % (cha.latitude or
                                                   sta.latitude))
                out.append("* LONGITUDE   : %s" % (cha.longitude or
                                                   sta.longitude))
                out.append("* ELEVATION   : %s" % (cha.elevation or
                                                   sta.elevation))
                out.append("* DEPTH       : %s" % cha.depth)
                # DIP in SACPZ served by IRIS SACPZ web service is
                # systematically different from the StationXML entries.
                # It is defined as an incidence angle (like done in SAC for
                # sensor orientation), rather than an actual dip.
                # Add '(SEED)' to clarify that we adhere to SEED convention
                out.append("* DIP (SEED)  : %s" % cha.dip)
                out.append("* AZIMUTH     : %s" % cha.azimuth)
                out.append("* SAMPLE RATE : %s" % cha.sample_rate)
                out.append("* INPUT UNIT  : M")
                out.append("* OUTPUT UNIT : %s" % sens.output_units)
                out.append("* INSTTYPE    : %s" % cha.sensor.type)
                out.append("* INSTGAIN    : %s (%s)" % (paz.stage_gain,
                                                        sens.input_units))
                out.append("* SENSITIVITY : %s (%s)" % (sens.value,
                                                        sens.input_units))
                out.append("* A0          : %s" % paz.normalization_factor)
                out.append("* " + "*" * 50)
                out.append(paz_to_sacpz_string(paz, sens))
                out.extend(["", ""])
    out = "\n".join(out) + "\n\n"
    try:
        file_or_file_object.write(out)
    except AttributeError:
        with open(file_or_file_object, "wt") as fh:
            fh.write(out)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
