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
from future.utils import native_str

import numpy as np

from obspy import UTCDateTime
from obspy.core import AttribDict
from obspy.core.inventory.response import paz_to_sacpz_string


def _write_sacpz(inventory, file_or_file_object):
    """
    Writes an inventory object to a SACPZ file.

    .. note::
        The channel DIP in the SACPZ comment fields is specified like defined
        by SEED (positive down from horizontal).

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.inventory.inventory.Inventory.write` method
        of an ObsPy :class:`~obspy.core.inventory.inventory.Inventory` object,
        call this instead.

    :type inventory: :class:`~obspy.core.inventory.inventory.Inventory`
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


# UTILITIES
def attach_paz(tr, paz_file, todisp=False, tovel=False, torad=False,
               tohz=False):
    '''
    Attach tr.stats.paz AttribDict to trace from SAC paz_file

    This is experimental code, taken from
    obspy.io.gse2.libgse2.attach_paz and adapted to the SAC-pole-zero
    conventions. Especially the conversion from velocity to
    displacement and vice versa is still under construction. It works
    but I cannot guarantee that the values are correct. For more
    information on the SAC-pole-zero format see:
    http://www.iris.edu/files/sac-manual/commands/transfer.html. For a
    useful discussion on polezero files and transfer functions in
    general see:
    http://seis-uk.le.ac.uk/equipment/downloads/data_management/\
seisuk_instrument_resp_removal.pdf
    Also bear in mind that according to the SAC convention for
    pole-zero files CONSTANT is defined as:
    digitizer_gain*seismometer_gain*A0. This means that it does not
    have explicit information on the digitizer gain and seismometer
    gain which we therefore set to 1.0.

    Attaches to a trace a paz AttribDict containing poles zeros and gain.

    :param tr: An ObsPy :class:`~obspy.core.trace.Trace` object
    :param paz_file: path to pazfile or file pointer
    :param todisp: change a velocity transfer function to a displacement
                   transfer function by adding another zero
    :param tovel: change a displacement transfer function to a velocity
                  transfer function by removing one 0,0j zero
    :param torad: change to radians
    :param tohz: change to Hertz

    >>> from obspy import Trace
    >>> import io
    >>> tr = Trace()
    >>> f = io.StringIO("""ZEROS 3
    ... -5.032 0.0
    ... POLES 6
    ... -0.02365 0.02365
    ... -0.02365 -0.02365
    ... -39.3011 0.
    ... -7.74904 0.
    ... -53.5979 21.7494
    ... -53.5979 -21.7494
    ... CONSTANT 2.16e18""")
    >>> attach_paz(tr, f,torad=True)
    >>> for z in tr.stats.paz['zeros']:
    ...     print("%.2f %.2f" % (z.real, z.imag))
    -31.62 0.00
    0.00 0.00
    0.00 0.00
    '''

    poles = []
    zeros = []

    if isinstance(paz_file, (str, native_str)):
        paz_file = open(paz_file, 'r')
        is_filename = True
    else:
        is_filename = False

    try:
        while True:
            line = paz_file.readline()
            if not line:
                break
            # lines starting with * are comments
            if line.startswith('*'):
                continue
            if line.find('ZEROS') != -1:
                a = line.split()
                noz = int(a[1])
                for _k in range(noz):
                    line = paz_file.readline()
                    a = line.split()
                    if line.find('POLES') != -1 or \
                       line.find('CONSTANT') != -1 or \
                       line.startswith('*') or not line:
                        while len(zeros) < noz:
                            zeros.append(complex(0, 0j))
                        break
                    else:
                        zeros.append(complex(float(a[0]), float(a[1])))

            if line.find('POLES') != -1:
                a = line.split()
                nop = int(a[1])
                for _k in range(nop):
                    line = paz_file.readline()
                    a = line.split()
                    if line.find('CONSTANT') != -1 or \
                       line.find('ZEROS') != -1 or \
                       line.startswith('*') or not line:
                        while len(poles) < nop:
                            poles.append(complex(0, 0j))
                        break
                    else:
                        poles.append(complex(float(a[0]), float(a[1])))
            if line.find('CONSTANT') != -1:
                a = line.split()
                # in the observatory this is the seismometer gain [muVolt/nm/s]
                # the A0_normalization_factor is hardcoded to 1.0
                constant = float(a[1])
    finally:
        if is_filename:
            paz_file.close()

    # To convert the velocity response to the displacement response,
    # multiplication with jw is used. This is equivalent to one more
    # zero in the pole-zero representation
    if todisp:
        zeros.append(complex(0, 0j))

    # To convert the displacement response to the velocity response,
    # division with jw is used. This is equivalent to one less zero
    # in the pole-zero representation
    if tovel:
        for i, zero in enumerate(list(zeros)):
            if zero == complex(0, 0j):
                zeros.pop(i)
                break
        else:
            raise Exception("Could not remove (0,0j) zero to change \
            displacement response to velocity response")

    # convert poles, zeros and gain in Hertz to radians
    if torad:
        tmp = [z * 2. * np.pi for z in zeros]
        zeros = tmp
        tmp = [p * 2. * np.pi for p in poles]
        poles = tmp
        # When extracting RESP files and SAC_PZ files
        # from a dataless SEED using the rdseed program
        # where the former is in Hz and the latter in radians,
        # there gains seem to be unaffected by this.
        # According to this document:
        # http://www.le.ac.uk/
        #         seis-uk/downloads/seisuk_instrument_resp_removal.pdf
        # the gain should also be converted when changing from
        # hertz to radians or vice versa. However, the rdseed programs
        # does not do this. I'm not entirely sure at this stage which one is
        # correct or if I have missed something. I've therefore decided
        # to leave it out for now, in order to stay compatible with the
        # rdseed program and the SAC program.
        # constant *= (2. * np.pi) ** 3

    # convert poles, zeros and gain in radian to Hertz
    if tohz:
        for i, z in enumerate(zeros):
            if abs(z) > 0.0:
                zeros[i] /= 2 * np.pi
        for i, p in enumerate(poles):
            if abs(p) > 0.0:
                poles[i] /= 2 * np.pi
        # constant /= (2. * np.pi) ** 3

    # fill up ObsPy Poles and Zeros AttribDict
    # In SAC pole-zero files CONSTANT is defined as:
    # digitizer_gain*seismometer_gain*A0

    tr.stats.paz = AttribDict()
    tr.stats.paz.seismometer_gain = 1.0
    tr.stats.paz.digitizer_gain = 1.0
    tr.stats.paz.poles = poles
    tr.stats.paz.zeros = zeros
    # taken from obspy.io.gse2.paz:145
    tr.stats.paz.sensitivity = tr.stats.paz.digitizer_gain * \
        tr.stats.paz.seismometer_gain
    tr.stats.paz.gain = constant


def attach_resp(tr, resp_file, todisp=False, tovel=False, torad=False,
                tohz=False):
    """
    Extract key instrument response information from a RESP file, which
    can be extracted from a dataless SEED volume by, for example, using
    the script obspy-dataless2resp or the rdseed program. At the moment,
    you have to determine yourself if the given response is for velocity
    or displacement and if the values are given in rad or Hz. This is
    still experimental code (see also documentation for
    :func:`obspy.io.sac.sacio.attach_paz`).
    Attaches to a trace a paz AttribDict containing poles, zeros, and gain.

    :param tr: An ObsPy :class:`~obspy.core.trace.Trace` object
    :param resp_file: path to RESP-file or file pointer
    :param todisp: change a velocity transfer function to a displacement
                   transfer function by adding another zero
    :param tovel: change a displacement transfer function to a velocity
                  transfer function by removing one 0,0j zero
    :param torad: change to radians
    :param tohz: change to Hertz

    >>> from obspy import Trace
    >>> import os
    >>> tr = Trace()
    >>> respfile = os.path.join(os.path.dirname(__file__), 'tests', 'data',
    ...                         'RESP.NZ.CRLZ.10.HHZ')
    >>> attach_resp(tr, respfile, torad=True, todisp=False)
    >>> for k in sorted(tr.stats.paz):  # doctest: +NORMALIZE_WHITESPACE
    ...     print(k)
    digitizer_gain
    gain
    poles
    seismometer_gain
    sensitivity
    t_shift
    zeros
    >>> print(tr.stats.paz.poles)  # doctest: +SKIP
    [(-0.15931644664884559+0.15931644664884559j),
     (-0.15931644664884559-0.15931644664884559j),
     (-314.15926535897933+202.31856689118268j),
     (-314.15926535897933-202.31856689118268j)]
    """
    if not hasattr(resp_file, 'write'):
        resp_filep = open(resp_file, 'r')
    else:
        resp_filep = resp_file

    zeros_pat = r'B053F10-13'
    poles_pat = r'B053F15-18'
    a0_pat = r'B053F07'
    sens_pat = r'B058F04'
    t_shift_pat = r'B057F08'
    t_shift = 0.0
    poles = []
    zeros = []
    while True:
        line = resp_filep.readline()
        if not line:
            break
        if line.startswith(a0_pat):
            a0 = float(line.split(':')[1])
        if line.startswith(sens_pat):
            sens = float(line.split(':')[1])
        if line.startswith(poles_pat):
            tmp = line.split()
            poles.append(complex(float(tmp[2]), float(tmp[3])))
        if line.startswith(zeros_pat):
            tmp = line.split()
            zeros.append(complex(float(tmp[2]), float(tmp[3])))
        if line.startswith(t_shift_pat):
            t_shift += float(line.split(':')[1])
    constant = a0 * sens

    if not hasattr(resp_file, 'write'):
        resp_filep.close()

    if torad:
        tmp = [z * 2. * np.pi for z in zeros]
        zeros = tmp
        tmp = [p * 2. * np.pi for p in poles]
        poles = tmp

    if todisp:
        zeros.append(complex(0, 0j))

    # To convert the displacement response to the velocity response,
    # division with jw is used. This is equivalent to one less zero
    # in the pole-zero representation
    if tovel:
        for i, zero in enumerate(list(zeros)):
            if zero == complex(0, 0j):
                zeros.pop(i)
                break
        else:
            raise Exception("Could not remove (0,0j) zero to change \
            displacement response to velocity response")

    # convert poles, zeros and gain in radian to Hertz
    if tohz:
        for i, z in enumerate(zeros):
            if abs(z) > 0.0:
                zeros[i] /= 2 * np.pi
        for i, p in enumerate(poles):
            if abs(p) > 0.0:
                poles[i] /= 2 * np.pi
        constant /= (2. * np.pi) ** 3

    # fill up ObsPy Poles and Zeros AttribDict
    # In SAC pole-zero files CONSTANT is defined as:
    # digitizer_gain*seismometer_gain*A0

    tr.stats.paz = AttribDict()
    tr.stats.paz.seismometer_gain = sens
    tr.stats.paz.digitizer_gain = 1.0
    tr.stats.paz.poles = poles
    tr.stats.paz.zeros = zeros
    # taken from obspy.io.gse2.paz:145
    tr.stats.paz.sensitivity = tr.stats.paz.digitizer_gain * \
        tr.stats.paz.seismometer_gain
    tr.stats.paz.gain = constant
    tr.stats.paz.t_shift = t_shift


if __name__ == "__main__":
    import doctest
    doctest.testmod()
