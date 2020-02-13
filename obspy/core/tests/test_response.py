#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the response handling.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
import os
import unittest
import warnings
from math import pi

import numpy as np
import scipy.interpolate
from matplotlib import rcParams

from obspy import UTCDateTime, read_inventory
from obspy.core.inventory.response import (
    _pitick2latex, PolesZerosResponseStage, PolynomialResponseStage, Response)
from obspy.core.util import MATPLOTLIB_VERSION
from obspy.core.util.misc import CatchOutput
from obspy.core.util.obspy_types import ComplexWithUncertainties
from obspy.core.util.testing import ImageComparison
from obspy.signal.invsim import evalresp
from obspy.io.xseed import Parser


class ResponseTestCase(unittest.TestCase):
    """
    Tests the for :class:`~obspy.core.inventory.response.Response` class.
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.nperr = np.geterr()
        np.seterr(all='ignore')

    def tearDown(self):
        np.seterr(**self.nperr)

    def test_evalresp_with_output_from_seed(self):
        """
        The StationXML file has been converted to SEED with the help of a tool
        provided by IRIS:

        https://seiscode.iris.washington.edu/projects/stationxml-converter
        """
        t_samp = 0.05
        nfft = 16384

        # Test for different output units.
        units = ["DISP", "VEL", "ACC"]
        filenames = ["IRIS_single_channel_with_response", "XM.05", "AU.MEEK"]

        for filename in filenames:
            xml_filename = os.path.join(self.data_dir,
                                        filename + os.path.extsep + "xml")
            seed_filename = os.path.join(self.data_dir,
                                         filename + os.path.extsep + "seed")

            p = Parser(seed_filename)

            # older systems don't like an end date in the year 2599
            t_ = UTCDateTime(2030, 1, 1)
            if p.blockettes[50][0].end_effective_date > t_:
                p.blockettes[50][0].end_effective_date = None
            if p.blockettes[52][0].end_date > t_:
                p.blockettes[52][0].end_date = None

            resp_filename = p.get_resp()[0][-1]

            inv = read_inventory(xml_filename)

            network = inv[0].code
            station = inv[0][0].code
            location = inv[0][0][0].location_code
            channel = inv[0][0][0].code
            date = inv[0][0][0].start_date

            for unit in units:
                resp_filename.seek(0, 0)

                seed_response, seed_freq = evalresp(
                    t_samp, nfft, resp_filename, date=date, station=station,
                    channel=channel, network=network, locid=location,
                    units=unit, freq=True)

                xml_response, xml_freq = \
                    inv[0][0][0].response.get_evalresp_response(t_samp, nfft,
                                                                output=unit)

                self.assertTrue(np.allclose(seed_freq, xml_freq, rtol=1E-5))
                self.assertTrue(np.allclose(seed_response, xml_response,
                                            rtol=1E-5))

                # also test getting response for a set of discrete frequencies
                indices = (-2, 0, -1, 1, 2, 20, -30, -100)
                freqs = [seed_freq[i_] for i_ in indices]
                response = inv[0][0][0].response
                got = response.get_evalresp_response_for_frequencies(
                    freqs, output=unit)
                expected = [seed_response[i_] for i_ in indices]
                np.testing.assert_allclose(got, expected, rtol=1E-5)

    def test_pitick2latex(self):
        self.assertEqual(_pitick2latex(3 * pi / 2), r'$\frac{3\pi}{2}$')
        self.assertEqual(_pitick2latex(2 * pi / 2), r'$\pi$')
        self.assertEqual(_pitick2latex(1 * pi / 2), r'$\frac{\pi}{2}$')
        self.assertEqual(_pitick2latex(0 * pi / 2), r'$0$')
        self.assertEqual(_pitick2latex(-1 * pi / 2), r'$-\frac{\pi}{2}$')
        self.assertEqual(_pitick2latex(-2 * pi / 2), r'$-\pi$')
        self.assertEqual(_pitick2latex(0.5), r'0.500')
        self.assertEqual(_pitick2latex(3 * pi + 0.01), r'9.43')
        self.assertEqual(_pitick2latex(30 * pi + 0.01), r'94.3')
        self.assertEqual(_pitick2latex(300 * pi + 0.01), r'942.')
        self.assertEqual(_pitick2latex(3000 * pi + 0.01), r'9.42e+03')

    def test_response_plot(self):
        """
        Tests the response plot.
        """
        # Bug in matplotlib 1.4.0 - 1.4.x:
        # See https://github.com/matplotlib/matplotlib/issues/4012
        reltol = 1.0
        if [1, 4, 0] <= MATPLOTLIB_VERSION <= [1, 5, 0]:
            reltol = 2.0

        resp = read_inventory()[0][0][0].response
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            with ImageComparison(self.image_dir, "response_response.png",
                                 reltol=reltol) as ic:
                rcParams['savefig.dpi'] = 72
                resp.plot(0.001, output="VEL", start_stage=1, end_stage=3,
                          outfile=ic.name)

    def test_response_plot_degrees(self):
        """
        Tests the response plot in degrees.
        """
        # Bug in matplotlib 1.4.0 - 1.4.x:
        # See https://github.com/matplotlib/matplotlib/issues/4012
        reltol = 1.0
        if [1, 4, 0] <= MATPLOTLIB_VERSION <= [1, 5, 0]:
            reltol = 2.0

        resp = read_inventory()[0][0][0].response
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            with ImageComparison(self.image_dir,
                                 "response_response_degrees.png",
                                 reltol=reltol) as ic:
                rcParams['savefig.dpi'] = 72
                resp.plot(0.001, output="VEL", start_stage=1, end_stage=3,
                          plot_degrees=True, outfile=ic.name)

    def test_segfault_after_error_handling(self):
        """
        Many functions in evalresp call `error_return()` which uses longjmp()
        to jump to some previously set state.

        ObsPy calls some evalresp functions directly so evalresp cannot call
        setjmp(). In that case longjmp() jumps to an undefined location, most
        likely resulting in a segfault.

        This test tests a workaround for this issue.

        As long as it does not segfault the test is doing alright.
        """
        filename = os.path.join(self.data_dir,
                                "TM.SKLT.__.BHZ_faulty_response.xml")
        inv = read_inventory(filename)

        t_samp = 0.05
        nfft = 256

        with CatchOutput():
            self.assertRaises(ValueError,
                              inv[0][0][0].response.get_evalresp_response,
                              t_samp, nfft, output="DISP")

    def test_custom_types_init(self):
        """
        Test initializations that involve custom decimal types like
        `ComplexWithUncertainties`.
        """
        # initializing poles / zeros from native types should work
        poles = [1 + 1j, 1, 1j]
        zeros = [2 + 3j, 2, 3j]
        stage = PolesZerosResponseStage(
            1, 1, 1, "", "", "LAPLACE (HERTZ)", 1, zeros, poles)
        self.assertEqual(type(stage.zeros[0]), ComplexWithUncertainties)
        self.assertEqual(type(stage.poles[0]), ComplexWithUncertainties)
        self.assertEqual(stage.poles, poles)
        self.assertEqual(stage.zeros, zeros)

    def test_response_list_stage(self):
        """
        This is quite rare but it happens.
        """
        inv = read_inventory(os.path.join(self.data_dir, "IM_IL31__BHZ.xml"))

        sampling_rate = 40.0
        t_samp = 1.0 / sampling_rate
        nfft = 100

        cpx_response, freq = inv[0][0][0].response.get_evalresp_response(
            t_samp=t_samp, nfft=nfft, output="VEL", start_stage=None,
            end_stage=None)

        # Cut of the zero frequency.
        cpx_response = cpx_response[1:]

        amp = np.abs(cpx_response)
        phase = np.angle(cpx_response)
        freq = freq[1:]

        # The expected output goes from 1 to 20 Hz - its somehow really hard
        # to get evalresp to produce results for the desired frequencies so
        # I just gave up on it.
        exp_f, exp_amp, exp_ph = np.loadtxt(os.path.join(
            self.data_dir, "expected_response_IM_IL31__BHZ.txt")).T
        # Interpolate.
        exp_amp = scipy.interpolate.InterpolatedUnivariateSpline(
            exp_f, exp_amp, k=3)(freq)
        exp_ph = scipy.interpolate.InterpolatedUnivariateSpline(
            exp_f, exp_ph, k=3)(freq)
        exp_ph = np.deg2rad(exp_ph)

        # The output is not exactle the same as ObsPy performs a different
        # but visually quite a bit better interpolation.
        np.testing.assert_allclose(amp, exp_amp, rtol=1E-3)
        np.testing.assert_allclose(phase, exp_ph, rtol=1E-3)

    def test_response_list_raises_error_if_out_of_range(self):
        """
        If extrpolating a lot it should raise an error.
        """
        inv = read_inventory(os.path.join(self.data_dir, "IM_IL31__BHZ.xml"))

        # The true sampling rate is 40 - this will thus request data that is
        # too high frequent and thus cannot be extracted from the response
        # list.
        sampling_rate = 45.0
        t_samp = 1.0 / sampling_rate
        nfft = 100

        with self.assertRaises(ValueError) as e:
            inv[0][0][0].response.get_evalresp_response(
                t_samp=t_samp, nfft=nfft, output="VEL", start_stage=None,
                end_stage=None)

        self.assertEqual(
            str(e.exception),
            "Cannot calculate the response as it contains a response list "
            "stage with frequencies only from -0.0096 - 20.0096 Hz. You are "
            "requesting a response from 0.4500 - 22.5000 Hz.")

    def test_response_with_no_units_in_stage_1(self):
        """
        ObsPy has some heuristics to deal with this particular degenerate case.
        Test it here.
        """
        inv = read_inventory(os.path.join(
            self.data_dir, "stationxml_no_units_in_stage_1.xml"))
        r = inv[0][0][0].response

        # The units should already have been fixed from reading the StationXML
        # files...
        self.assertEqual(r.response_stages[0].input_units, "M/S")
        self.assertEqual(r.response_stages[0].input_units_description,
                         "Meters per second")
        self.assertEqual(r.response_stages[0].output_units, "V")
        self.assertEqual(r.response_stages[0].output_units_description,
                         "VOLTS")

        # We have to set the units to None here as there is some other logic in
        # reading the StationXML files that sets them based on other units...
        r.response_stages[0].input_units = None
        r.response_stages[0].output_units = None

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = r.get_evalresp_response_for_frequencies(
                np.array([0.5, 1.0, 2.0]), output="DISP")

        self.assertEqual(len(w), 2)
        self.assertEqual(
            w[0].message.args[0],
            "Set the input units of stage 1 to the overall input units.")
        self.assertEqual(
            w[1].message.args[0],
            "Set the output units of stage 1 to the input units of stage 2.")

        # Values compared to evalresp output from RESP file - might not be
        # right but it does guarantee that ObsPy behaves like evalresp - be
        # that a good thing or a bad thing.
        np.testing.assert_allclose(
            out, [0 + 9869.2911771081963j, 0 + 19738.582354216393j,
                  0 + 39477.164708432785j])

    def test_resp_from_paz_setting_zeros_poles_and_sensitivity(self):
        poles = [1 + 2j, 1 - 2j, 2 + 1j, 2 - 1j]
        zeros = [0, 0, 5]
        sensitivity = 1201. * (2 ** 26 / 40.)
        # a0 normalization factor at 1Hz for these values
        normalization = np.prod(2 * pi * 1j - np.array(zeros))
        normalization /= np.prod(2 * pi * 1j - np.array(poles))
        normalization = np.abs(normalization)
        resp = Response.from_paz(zeros, poles, sensitivity)
        r_zeros = resp.response_stages[0].zeros
        r_poles = resp.response_stages[0].poles
        r_stage_gain = resp.response_stages[0].stage_gain
        r_sens = resp.instrument_sensitivity.value
        np.testing.assert_array_equal(r_zeros, zeros)
        np.testing.assert_array_equal(r_poles, poles)
        np.testing.assert_equal(r_stage_gain, sensitivity)
        np.testing.assert_allclose(r_sens / normalization, sensitivity, atol=0,
                                   rtol=1e-8)

    def test_resp_from_paz_loading_vs_evalresp(self):
        zeros = [0., 0.]
        poles = [-8.443 + 1.443j, -8.443 - 1.443j]
        filename = os.path.join(self.data_dir,
                                'RESP.XX.NS306..SHZ.GS13.1.2180')
        resp_er = read_inventory(filename)[0][0][0].response
        loaded_resp = resp_er.get_evalresp_response(.1, 2**6, output='VEL')
        # The optional kwargs are the same as those being set in the RESP file.
        resp = Response.from_paz(zeros=zeros, poles=poles, stage_gain=22.0,
                                 stage_gain_frequency=5.0,
                                 normalization_frequency=5.0,
                                 normalization_factor=1.070401)
        paz_resp = resp.get_evalresp_response(.1, 2**6, output='VEL')
        np.testing.assert_allclose(paz_resp, loaded_resp)

    def test_str_method_of_the_polynomial_response_stage(self):
        # First with gain and gain frequency.
        self.assertEqual(str(PolynomialResponseStage(
            stage_sequence_number=2,
            stage_gain=12345.0,
            stage_gain_frequency=1.0,
            input_units="PA",
            input_units_description="Pascal",
            output_units="COUNTS",
            output_units_description="digital_counts",
            frequency_lower_bound=1.0,
            frequency_upper_bound=2.0,
            approximation_lower_bound=3.0,
            approximation_upper_bound=4.0,
            maximum_error=1.5,
            coefficients=[1.0, 2.0, 3.0],
            approximation_type="MACLAURIN",
            decimation_input_sample_rate=1.0,
            decimation_factor=2,
            decimation_offset=3.0,
            decimation_delay=4.0,
            decimation_correction=True)),
            "Response type: PolynomialResponseStage, "
            "Stage Sequence Number: 2\n"
            "\tFrom PA (Pascal) to COUNTS (digital_counts)\n"
            "\tStage gain: 12345.0, defined at 1.00 Hz\n"
            "\tDecimation:\n"
            "\t\tInput Sample Rate: 1.00 Hz\n"
            "\t\tDecimation Factor: 2\n"
            "\t\tDecimation Offset: 3\n"
            "\t\tDecimation Delay: 4.00\n"
            "\t\tDecimation Correction: 1.00\n"
            "\tPolynomial approximation type: MACLAURIN\n"
            "\tFrequency lower bound: 1.0\n"
            "\tFrequency upper bound: 2.0\n"
            "\tApproximation lower bound: 3.0\n"
            "\tApproximation upper bound: 4.0\n"
            "\tMaximum error: 1.5\n"
            "\tNumber of coefficients: 3"
        )

        # Now only the very minimum.
        self.assertEqual(str(PolynomialResponseStage(
            stage_sequence_number=4,
            stage_gain=None,
            stage_gain_frequency=None,
            input_units=None,
            input_units_description=None,
            output_units=None,
            output_units_description=None,
            frequency_lower_bound=None,
            frequency_upper_bound=None,
            approximation_lower_bound=None,
            approximation_upper_bound=None,
            maximum_error=None,
            coefficients=[],
            approximation_type="MACLAURIN")),
            "Response type: PolynomialResponseStage, "
            "Stage Sequence Number: 4\n"
            "\tFrom UNKNOWN to UNKNOWN\n"
            "\tStage gain: UNKNOWN, defined at UNKNOWN Hz\n"
            "\tPolynomial approximation type: MACLAURIN\n"
            "\tFrequency lower bound: None\n"
            "\tFrequency upper bound: None\n"
            "\tApproximation lower bound: None\n"
            "\tApproximation upper bound: None\n"
            "\tMaximum error: None\n"
            "\tNumber of coefficients: 0"
        )

    def test_get_sampling_rates(self):
        """
        Tests for the get_sampling_rates() method.
        """
        # Test for the default inventory.
        resp = read_inventory()[0][0][0].response
        self.assertEqual(
            resp.get_sampling_rates(),
            {1: {'decimation_factor': 1,
                 'input_sampling_rate': 200.0,
                 'output_sampling_rate': 200.0},
             2: {'decimation_factor': 1,
                 'input_sampling_rate': 200.0,
                 'output_sampling_rate': 200.0}})

        # Another, well behaved file.
        inv = read_inventory(os.path.join(self.data_dir, "AU.MEEK.xml"))
        self.assertEqual(
            inv[0][0][0].response.get_sampling_rates(),
            {1: {'decimation_factor': 1,
                 'input_sampling_rate': 600.0,
                 'output_sampling_rate': 600.0},
             2: {'decimation_factor': 1,
                 'input_sampling_rate': 600.0,
                 'output_sampling_rate': 600.0},
             3: {'decimation_factor': 1,
                 'input_sampling_rate': 600.0,
                 'output_sampling_rate': 600.0},
             4: {'decimation_factor': 3,
                 'input_sampling_rate': 600.0,
                 'output_sampling_rate': 200.0},
             5: {'decimation_factor': 10,
                 'input_sampling_rate': 200.0,
                 'output_sampling_rate': 20.0}})

        # This file lacks decimation attributes for the first two stages as
        # well as one of the later ones. These thus have to be inferred.
        inv = read_inventory(os.path.join(self.data_dir, "DK.BSD..BHZ.xml"))
        self.assertEqual(
            inv[0][0][0].response.get_sampling_rates(),
            {1: {'decimation_factor': 1,
                 'input_sampling_rate': 30000.0,
                 'output_sampling_rate': 30000.0},
             2: {'decimation_factor': 1,
                 'input_sampling_rate': 30000.0,
                 'output_sampling_rate': 30000.0},
             3: {'decimation_factor': 1,
                 'input_sampling_rate': 30000.0,
                 'output_sampling_rate': 30000.0},
             4: {'decimation_factor': 5,
                 'input_sampling_rate': 30000.0,
                 'output_sampling_rate': 6000.0},
             5: {'decimation_factor': 3,
                 'input_sampling_rate': 6000.0,
                 'output_sampling_rate': 2000.0},
             6: {'decimation_factor': 2,
                 'input_sampling_rate': 2000.0,
                 'output_sampling_rate': 1000.0},
             7: {'decimation_factor': 5,
                 'input_sampling_rate': 1000.0,
                 'output_sampling_rate': 200.0},
             8: {'decimation_factor': 2,
                 'input_sampling_rate': 200.0,
                 'output_sampling_rate': 100.0},
             9: {'decimation_factor': 1,
                 'input_sampling_rate': 100.0,
                 'output_sampling_rate': 100.0},
             10: {'decimation_factor': 5,
                  'input_sampling_rate': 100.0,
                  'output_sampling_rate': 20.0}})

    def test_response_calculation_paz_without_decimation(self):
        """
        This test files has two PAZ stages with no decimation attributes.

        Evalresp does not like this so we have to add dummy decimation
        attributes before calling it.
        """
        inv = read_inventory(os.path.join(self.data_dir, "DK.BSD..BHZ.xml"))
        np.testing.assert_allclose(
            inv[0][0][0].response.get_evalresp_response_for_frequencies(
                [0.1, 1.0, 10.0]),
            [6.27191825e+08 + 1.38925202e+08j,
             6.51826202e+08 + 1.28404787e+07j,
             2.00067263e+04 - 2.63711751e+03j])

    def test_regression_evalresp(self):
        """
        Regression test for an evalresp issue with a micropressure instrument.

        See #2171.
        """
        inv = read_inventory(os.path.join(self.data_dir, "IM_I53H1_BDF.xml"))
        self.assertEqual(
            inv[0][0][0].response.get_evalresp_response_for_frequencies([0.0]),
            0.0 + 0.0j)
        np.testing.assert_allclose(
            inv[0][0][0].response.get_evalresp_response_for_frequencies(
                [0.1, 1.0, 10.0]),
            [2.411908e+05 + 2.283852e+04j,
             2.445572e+05 - 2.480459e+03j,
             -2.455459e-01 + 4.888214e-02j], rtol=1e-6)

    def test_recalculate_overall_sensitivity(self):
        """
        Tests the recalculate_overall_sensitivity_method().

        This is not yet an exhaustive test as responses are complicated...
        """
        resp = read_inventory()[0][0][0].response
        np.testing.assert_allclose(
            resp.instrument_sensitivity.value,
            943680000.0)
        np.testing.assert_allclose(
            resp.instrument_sensitivity.frequency,
            0.02)
        # Recompute - it is not much different but a bit.
        resp.recalculate_overall_sensitivity(0.02)
        np.testing.assert_allclose(
            resp.instrument_sensitivity.value,
            943681500.0)
        np.testing.assert_allclose(
            resp.instrument_sensitivity.frequency,
            0.02)

        # There is some logic to automatically determine a suitable frequency.
        # Make sure this does something here.
        resp = read_inventory()[0][0][0].response
        resp.recalculate_overall_sensitivity()
        np.testing.assert_allclose(
            resp.instrument_sensitivity.value,
            957562105.3939067)
        np.testing.assert_allclose(
            resp.instrument_sensitivity.frequency,
            1.0)

        # Passing an integer also works. See #2338.
        resp = read_inventory()[0][0][0].response
        resp.recalculate_overall_sensitivity(1)
        np.testing.assert_allclose(
            resp.instrument_sensitivity.value,
            957562105.3939067)
        np.testing.assert_allclose(
            resp.instrument_sensitivity.frequency,
            1.0)


def suite():
    return unittest.makeSuite(ResponseTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
