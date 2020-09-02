# -*- coding: utf-8 -*-
import copy
import inspect
import io
import os
import unittest

from obspy import read_events
from obspy.core.event.base import QuantityError
from obspy.core.util.base import get_example_file, CatchAndAssertWarnings
from obspy.io.hyposat.core import (
    _pick_to_hyposat_phase_line, _write_hyposat_phases)


class HYPOSATTestCase(unittest.TestCase):
    """
    Test suite for obspy.io.hyposat.core
    """

    def setUp(self):
        self.path = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.catalog = read_events(
            get_example_file('nlloc.qml'), format='QUAKEML')
        self.file_expected = os.path.join(self.path, 'data', 'nlloc.in')
        # set up expected warnings
        msgs = [
            "Station code exceeds five characters, getting truncated: "
            "'ABCDEF'",
            "Using pick without station code: "
            "smi:local/a040ecfb-0d12-4c91-9e37-463f075a2ec6",
            "Using pick without phase hint: "
            "smi:local/a040ecfb-0d12-4c91-9e37-463f075a2ec6",
            "Using pick without station code: "
            "smi:local/a040ecfb-0d12-4c91-9e37-463f075a2ec6",
            "Phase hint exceeds eight characters, getting truncated: "
            "'pPSPSPSPS'"]
        self.expected_warnings = [(UserWarning, msg) for msg in msgs]

        # make some artificial changes to improve tests
        # add a later S pick as first pick to check automatic sorting
        picks = self.catalog[0].picks
        pick = copy.deepcopy(picks[1])
        pick.phase_hint = 'S'
        pick.time = pick.time + 6
        picks.insert(0, pick)
        # modify one P pick to Pg and add a second pick as Pn (since single P
        # picks should get -- optionally -- renamed to P1)
        pick = picks[3]
        pick.phase_hint = 'Pg'
        pick = copy.deepcopy(pick)
        pick.phase_hint = 'Pn'
        pick.time = pick.time + 2
        picks.insert(0, pick)
        # add a pick with a long station name, emitting a warning
        pick = picks[-1]
        pick.waveform_id.station_code = 'ABCDEF'
        # fill in some infos that are not present in the example data we read
        picks[0].backazimuth = 0.02345
        picks[1].backazimuth = 359.12345
        picks[0].backazimuth_errors = QuantityError(uncertainty=2.4)
        # make some assymetric uncertainties
        picks[0].time_errors.lower_uncertainty = 0.12
        picks[0].time_errors.upper_uncertainty = 0.34
        # add pick without station code and phase hint
        pick = copy.deepcopy(picks[-1])
        pick.time = pick.time + 2.3
        pick.waveform_id.station_code = None
        pick.phase_hint = None
        pick.horizontal_slowness_errors = QuantityError(uncertainty=1.2345)
        picks.append(pick)
        # add pick with overlong phase hint
        pick = copy.deepcopy(picks[-1])
        pick.time = pick.time + 2.3
        pick.phase_hint = 'pPSPSPSPS'
        pick.horizontal_slowness = 23.12345
        picks.append(pick)

    def test_write_hyposat(self):
        """
        Test HYPOSAT phases writing routine
        """
        with io.BytesIO() as buf:
            with CatchAndAssertWarnings(expected=self.expected_warnings):
                _write_hyposat_phases(
                    self.catalog, buf, hyposat_rename_first_onsets=True)
            buf.seek(0)
            got = buf.read()
        with open(self.file_expected, 'rb') as fh:
            expected = fh.read()
        self.assertEqual(got, expected)

    def test_write_hyposat_via_plugin(self):
        """
        Test writing HYPOSAT phases file via plugin
        """
        with io.BytesIO() as buf:
            with CatchAndAssertWarnings(expected=self.expected_warnings):
                _write_hyposat_phases(
                    self.catalog, buf, hyposat_rename_first_onsets=True)
            buf.seek(0)
            expected = buf.read()
        with io.BytesIO() as buf:
            with CatchAndAssertWarnings(expected=self.expected_warnings):
                self.catalog.write(
                    buf, format='HYPOSAT_PHASES',
                    hyposat_rename_first_onsets=True)
            buf.seek(0)
            got = buf.read()
        self.assertEqual(got, expected)

    def test_pick_to_hyposat_phase_line(self):
        """
        Test converting Pick to HYPOSAT phase line
        """
        got = _pick_to_hyposat_phase_line(self.catalog[0].picks[0])
        expected = (
            'HM05  Pn       2006 07 15 17 21 22.640 0.120   0.02  2.40'
            '                                                          0.340')
        self.assertEqual(got, expected)


def suite():
    return unittest.makeSuite(HYPOSATTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
