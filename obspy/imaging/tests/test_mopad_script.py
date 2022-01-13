# -*- coding: utf-8 -*-
"""
The obspy-mopad script test suite.
"""
import io
import os
from itertools import product, zip_longest

import numpy as np
import pytest

from obspy.core.util.misc import CatchOutput
from obspy.imaging.scripts.mopad import main as obspy_mopad


class TestMopad:
    """
    Test cases for obspy-mopad script.
    """
    path = os.path.join(os.path.dirname(__file__), 'images')
    mt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]

    def test_script_convert_type_sdr(self):
        with CatchOutput() as out:
            obspy_mopad(['convert', '--fancy', '-t', 'sdr',
                         ','.join(str(x) for x in self.mt)])
        expected = '''
Fault plane 1: strike =  77°, dip =  89°, slip-rake = -141°
Fault plane 2: strike = 346°, dip =  51°, slip-rake =   -1°
'''
        result = out.stdout[:-1]
        assert expected == result

    def test_script_convert_type_tensor(self):
        with CatchOutput() as out:
            obspy_mopad(['convert', '--fancy', '-t', 't',
                         ','.join(str(x) for x in self.mt)])
        expected = r'''
   Full moment tensor in NED-coordinates:

  /  0.91  1.78 -1.55 \
  |  1.78 -0.89  0.47  |
  \ -1.55  0.47 -0.02 /

'''
        assert expected == out.stdout

    def test_script_convert_type_tensor_large(self):
        with CatchOutput() as out:
            obspy_mopad(['convert', '--fancy', '-t', 't',
                         ','.join(str(x * 100) for x in self.mt)])
        expected = r'''
   Full moment tensor in NED-coordinates:

  /  0.51  1.00 -0.87 \
  |  1.00 -0.50  0.26  |   x  178.000000
  \ -0.87  0.26 -0.01 /

'''
        assert expected == out.stdout

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
            assert len(exp) == len(actual)
            for i, (e, a) in enumerate(zip(exp, actual)):
                msg = '%d: %f != %f in %s -> %s conversion' % (i, e, a,
                                                               insys, outsys)
                assert round(abs(e-a), 7) == 0, msg

    def test_script_convert_vector(self):
        with CatchOutput() as out:
            obspy_mopad(['convert', '-v', 'NED', 'NED',
                         ','.join(str(x) for x in self.mt)])
        expected = str(self.mt) + '\n'
        assert expected == out.stdout

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
        result = out.stdout[:-1]
        assert expected == result

    #
    # obspy-mopad gmt
    #

    def compare_gmt(self, exp_file, *args):
        """
        Helper function that runs GMT and compares results.
        """
        with CatchOutput() as out:
            obspy_mopad(['gmt'] + list(args) +
                        [','.join(str(x) for x in self.mt)])

        expected = os.path.join(self.path, exp_file)

        # Test headers
        with open(expected, 'rt') as expf:
            bio = out.stdout
            # expf.read().splitlines() differs to expf.readlines() ?!?!?!
            for exp_line, out_line in zip_longest(expf.read().splitlines(),
                                                  bio.splitlines(),
                                                  fillvalue=''):
                if exp_line.startswith('>') or out_line.startswith('>'):
                    assert exp_line == out_line, \
                                     'Headers do not match!'

        # Test actual data
        exp_data = np.genfromtxt(expected, comments='>')
        with io.BytesIO(out.stdout.encode('utf-8')) as bio:
            out_data = np.genfromtxt(bio, comments='>')
        assert exp_data.shape == out_data.shape, \
               'Data does not match!'
        assert np.allclose(exp_data, out_data), \
               'Data does not match!'

    def test_script_gmt_fill(self):
        self.compare_gmt('mopad_fill.gmt',
                         '-t', 'fill',
                         '--scaling', '2', '--color1', '3', '--color2', '5')

    def test_script_gmt_lines(self):
        self.compare_gmt('mopad_lines.gmt',
                         '-t', 'lines',
                         '--scaling', '2', '--color1', '3', '--color2', '5')

    def test_script_gmt_lines_stereo(self):
        self.compare_gmt('mopad_lines_stereo.gmt',
                         '-t', 'lines',
                         '--scaling', '2', '--color1', '3', '--color2', '5',
                         '--projection', 'stereo')

    def test_script_gmt_lines_ortho(self):
        self.compare_gmt('mopad_lines_ortho.gmt',
                         '-t', 'lines',
                         '--scaling', '2', '--color1', '3', '--color2', '5',
                         '--projection', 'ortho')

    def test_script_gmt_lines_lambo(self):
        self.compare_gmt('mopad_lines_lambo.gmt',
                         '-t', 'lines',
                         '--scaling', '2', '--color1', '3', '--color2', '5',
                         '--projection', 'stereo')

    def test_script_gmt_event(self):
        self.compare_gmt('mopad_ev.gmt',
                         '-t', 'ev',
                         '-r', '3')

    @pytest.mark.skip('Currently broken until further review.')
    def test_script_plot(self, image_path):
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
            with CatchOutput() as out:
                obspy_mopad(['plot',
                             '--output-file', image_path,
                             '--input-system', 'USE',
                             '--tension-color', 'b',
                             '--pressure-color', 'w',
                             '--lines', '1', 'k', '1',
                             '--nodals', '1', 'k', '1',
                             '--size', '2.54', '--quality', '200',
                             '--',
                             ','.join(str(x) for x in mt)])
            assert '' == out.stdout

        if messages:
            pytest.fail(messages)
