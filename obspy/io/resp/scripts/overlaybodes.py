#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Overlay bode plots from multiple responses
rdseed -R RESP files
As well as camparing to NRL resp files: 1 dl, 1 sensor.

:copyright:
    Lloyd Carothers IRIS/PASSCAL, 2016
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.core import inventory
from obspy.io.resp.nrl import NRL
from os.path import basename


def add_nrl_to_fig(fig, sensor_short, dl_short, sr, gain):
    nrl = NRL(local=False)
    sensor_resp = nrl.sensor_from_short(sensor_short)
    dl_resp = nrl.datalogger_from_short(dl_short, gain, sr)
    inv_resp = inventory.response.response_from_resps(sensor_resp,
                                                      dl_resp,
                                                      frequency=None,
                                                      sr=sr)
    label = '{} {}'.format(sensor_short, dl_short)
    print(inv_resp.get_sacpz())
    # First plot no figure to pass
    if fig is None:
        return inv_resp.plot(MIN_FREQ, UNIT, label=label, show=False)
    else:
        return inv_resp.plot(MIN_FREQ, UNIT, label=label, show=False,
                             axes=fig.axes)


def add_resp_to_fig(fig, resp_file):
    resp_data = open(resp_file).read()
    resp = inventory.response.response_from_resp(resp_data)
    label = basename(resp_file)
    print(resp.get_sacpz())
    if fig is None:
        return resp.plot(MIN_FREQ, UNIT, label=label, show=False)
    else:
        return resp.plot(MIN_FREQ, UNIT, label=label, show=False,
                         axes=fig.axes)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    VERSION = '2016.309'
    parser = ArgumentParser(description='Shows bode plot responses overlayed')
    parser.add_argument('-nrl', dest='nrl', nargs=4,
                        required=False, action='append',
                        help='sensor_nick dl_nick sr gain')
    parser.add_argument('-r', dest='resp_file',
                        required=False, action='append')
    parser.add_argument('-u', '--unit', dest='unit', default='VEL')
    parser.add_argument('-min_freq', dest='min_freq', default=.001)
    args = parser.parse_args()

    global MIN_FREQ, UNIT
    MIN_FREQ = args.min_freq
    UNIT = args.unit

    fig = None
    for sn_nick, dl_nick, sr, gain in args.nrl:
        print(sn_nick, dl_nick, sr, gain)
        fig = add_nrl_to_fig(fig, sn_nick, dl_nick, float(sr), gain)
    for filename in args.resp_file:
        print(filename)
        fig = add_resp_to_fig(fig, filename)

    # Fix broken legend
    fig.axes[0].legend(*fig.axes[0].get_legend_handles_labels())
    fig.axes[0].legend(loc='lower center')
    # Change line style so very similar response can be seen
    for i, line in enumerate(fig.axes[0].lines[::2]):
        line.set_dashes([3, 6-i])
    for i, line in enumerate(fig.axes[1].lines[::2]):
        line.set_dashes([3, 6-i])

    plt.show()
