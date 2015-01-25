# -*- coding: utf-8 -*-
"""
The obspy-mopad script test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.core.util.misc import CatchOutput
from obspy.core.util.testing import ImageComparison, ImageComparisonException
from obspy.core.util.decorator import skip
from obspy.imaging.scripts.mopad import main as obspy_mopad

import numpy as np

import io
from itertools import product, zip_longest
import os
import unittest


class MopadTestCase(unittest.TestCase):
    """
    Test cases for obspy-mopad script.
    """

    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'images')
        self.mt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]

    #
    # obspy-mopad convert
    #

    def test_script_convert_type_sdr(self):
        with CatchOutput() as out:
            obspy_mopad(['convert', '--fancy', '-t', 'sdr',
                         ','.join(str(x) for x in self.mt)])

        expected = '''
Fault plane 1: strike =  77°, dip =  89°, slip-rake = -141°
Fault plane 2: strike = 346°, dip =  51°, slip-rake =   -1°

'''

        self.assertEqual(expected, out.stdout.decode("utf-8"))

    def test_script_convert_type_tensor(self):
        with CatchOutput() as out:
            obspy_mopad(['convert', '--fancy', '-t', 't',
                         ','.join(str(x) for x in self.mt)])

        expected = '''
   Full moment tensor in NED-coordinates:

  /  0.91  1.78 -1.55 \\
  |  1.78 -0.89  0.47  |
  \ -1.55  0.47 -0.02 /

'''

        self.assertEqual(expected, out.stdout.decode("utf-8"))

    def test_script_convert_type_tensor_large(self):
        with CatchOutput() as out:
            obspy_mopad(['convert', '--fancy', '-t', 't',
                         ','.join(str(x * 100) for x in self.mt)])

        expected = '''
   Full moment tensor in NED-coordinates:

  /  0.51  1.00 -0.87 \\
  |  1.00 -0.50  0.26  |   x  178.000000
  \ -0.87  0.26 -0.01 /

'''

        self.assertEqual(expected, out.stdout.decode("utf-8"))

    def test_script_convert_basis(self):
        expected = [
            (0.91, -0.89, -0.02, 1.78, -1.55, 0.47),
            (-0.02, 0.91, -0.89, -1.55, -0.47, -1.78),
            (-0.89, 0.91, -0.02, 1.78, -0.47, 1.55),
            (0.91, -0.89, -0.02, -1.78, 1.55, 0.47),
            (-0.89, -0.02, 0.91, -0.47, 1.78, 1.55),
            (0.91, -0.89, -0.02, 1.78, -1.55, 0.47),
            (-0.02, -0.89, 0.91, -0.47, -1.55, -1.78),
            (-0.89, -0.02, 0.91, 0.47, -1.78, 1.55),
            (-0.89, 0.91, -0.02, 1.78, -0.47, 1.55),
            (-0.02, -0.89, 0.91, -0.47, -1.55, -1.78),
            (0.91, -0.89, -0.02, 1.78, -1.55, 0.47),
            (-0.89, 0.91, -0.02, -1.78, 0.47, 1.55),
            (0.91, -0.89, -0.02, -1.78, 1.55, 0.47),
            (-0.02, 0.91, -0.89, 1.55, -0.47, 1.78),
            (-0.89, 0.91, -0.02, -1.78, -0.47, -1.55),
            (0.91, -0.89, -0.02, 1.78, -1.55, 0.47)
        ]

        for exp, (insys, outsys) in zip(expected,
                                        product(['NED', 'USE', 'XYZ', 'NWU'],
                                                repeat=2)):

            with CatchOutput() as out:
                obspy_mopad(['convert', '-b', insys, outsys,
                             ','.join(str(x) for x in self.mt)])

            actual = eval(out.stdout)
            self.assertEqual(len(exp), len(actual))
            for i, (e, a) in enumerate(zip(exp, actual)):
                msg = '%d: %f != %f in %s -> %s conversion' % (i, e, a,
                                                               insys, outsys)
                self.assertAlmostEqual(e, a, msg=msg)

    def test_script_convert_vector(self):
        with CatchOutput() as out:
            obspy_mopad(['convert', '-v', 'NED', 'NED',
                         ','.join(str(x) for x in self.mt)])

        expected = str(self.mt) + '\n'

        self.assertEqual(expected, out.stdout.decode("utf-8"))

    #
    # obspy-mopad decompose
    #

    def test_script_decompose(self):
        with CatchOutput() as out:
            obspy_mopad(['decompose', '-y', ','.join(str(x) for x in self.mt)])

        expected = '''
Scalar Moment: M0 = 2.61206 Nm (Mw = -5.8)
Moment Tensor: Mnn =  0.091,  Mee = -0.089, Mdd = -0.002,
               Mne =  0.178,  Mnd = -0.155, Med =  0.047    [ x 10 ]


Fault plane 1: strike =  77°, dip =  89°, slip-rake = -141°
Fault plane 2: strike = 346°, dip =  51°, slip-rake =   -1°

'''

        self.assertEqual(expected, out.stdout.decode("utf-8"))

    #
    # obspy-mopad gmt
    #

    def compareGMT(self, exp_file, *args):
        """
        Helper function that runs GMT and compares results.
        """
        with CatchOutput() as out:
            obspy_mopad(['gmt'] + list(args) +
                        [','.join(str(x) for x in self.mt)])

        expected = os.path.join(self.path, exp_file)

        # Test headers
        with open(expected, 'rb') as expf:
            with io.BytesIO(out.stdout) as bio:
                for exp_line, out_line in zip_longest(expf.readlines(),
                                                      bio.readlines(),
                                                      fillvalue=''):
                    if exp_line.startswith(b'>') or out_line.startswith(b'>'):
                        self.assertEqual(exp_line, out_line,
                                         msg='Headers do not match!')

        # Test actual data
        exp_data = np.genfromtxt(expected, comments='>')
        with io.BytesIO(out.stdout) as bio:
            out_data = np.genfromtxt(bio, comments='>')
        self.assertEqual(exp_data.shape, out_data.shape,
                         msg='Data does not match!')
        self.assertTrue(np.allclose(exp_data, out_data),
                        msg='Data does not match!')

    def test_script_gmt_fill(self):
        self.compareGMT('mopad_fill.gmt',
                        '-t', 'fill',
                        '--scaling', '2', '--color1', '3', '--color2', '5')

    def test_script_gmt_lines(self):
        self.compareGMT('mopad_lines.gmt',
                        '-t', 'lines',
                        '--scaling', '2', '--color1', '3', '--color2', '5')

    def test_script_gmt_lines_stereo(self):
        self.compareGMT('mopad_lines_stereo.gmt',
                        '-t', 'lines',
                        '--scaling', '2', '--color1', '3', '--color2', '5',
                        '--projection', 'stereo')

    def test_script_gmt_lines_ortho(self):
        self.compareGMT('mopad_lines_ortho.gmt',
                        '-t', 'lines',
                        '--scaling', '2', '--color1', '3', '--color2', '5',
                        '--projection', 'ortho')

    def test_script_gmt_lines_lambo(self):
        self.compareGMT('mopad_lines_lambo.gmt',
                        '-t', 'lines',
                        '--scaling', '2', '--color1', '3', '--color2', '5',
                        '--projection', 'stereo')

    def test_script_gmt_event(self):
        self.compareGMT('mopad_ev.gmt',
                        '-t', 'ev',
                        '-r', '3')

    #
    # obspy-mopad plot
    #

    @skip('Currently broken until further review.')
    def test_script_plot(self):
        # See test_Beachball:
        data = [
            [0.91, -0.89, -0.02, 1.78, -1.55, 0.47],
            [274, 13, 55],
            [130, 79, 98],
            [264.98, 45.00, -159.99],
            [160.55, 76.00, -46.78],
            [1.45, -6.60, 5.14, -2.67, -3.16, 1.36],
            [235, 80, 35],
            [138, 56, 168],
            [1, 1, 1, 0, 0, 0],
            [-1, -1, -1, 0, 0, 0],
            [1, -2, 1, 0, 0, 0],
            [1, -1, 0, 0, 0, 0],
            [1, -1, 0, 0, 0, -1],
            [179, 55, -78],
            [10, 42.5, 90],
            [10, 42.5, 92],
            [150, 87, 1],
            [0.99, -2.00, 1.01, 0.92, 0.48, 0.15],
            [5.24, -6.77, 1.53, 0.81, 1.49, -0.05],
            [16.578, -7.987, -8.592, -5.515, -29.732, 7.517],
            [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94],
        ]
        filenames = [
            'bb_sumatra_mt.png', 'bb_sumatra_np1.png', 'bb_sumatra_np2.png',
            'bb_19950128_np1.png', 'bb_19950128_np2.png', 'bb_20090102_mt.png',
            'bb_20090102_np1.png', 'bb-20090102-np2.png', 'bb_explosion.png',
            'bb_implosion.png', 'bb_clvd.png', 'bb_double_couple.png',
            'bb_lars.png', 'bb_geiyo_np1.png', 'bb_honshu_np1.png',
            'bb_honshu_np2.png', 'bb_tottori_np1.png', 'bb_20040905_1_mt.png',
            'bb_20040905_0_mt.png', 'bb_miyagi_mt.png', 'bb_chile_mt.png',
        ]

        messages = ''
        for mt, filename in zip(data, filenames):
            try:
                with ImageComparison(self.path, filename) as ic:
                    with CatchOutput() as out:
                        obspy_mopad(['plot',
                                     '--output-file', ic.name,
                                     '--input-system', 'USE',
                                     '--tension-color', 'b',
                                     '--pressure-color', 'w',
                                     '--lines', '1', 'k', '1',
                                     '--nodals', '1', 'k', '1',
                                     '--size', '2.54', '--quality', '200',
                                     '--',
                                     ','.join(str(x) for x in mt)])
                    self.assertEqual('', out.stdout)
            except ImageComparisonException as e:
                if ic.keep_output:
                    messages += str(e)
                else:
                    raise

        if messages:
            self.fail(messages)


def suite():
    return unittest.makeSuite(MopadTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
