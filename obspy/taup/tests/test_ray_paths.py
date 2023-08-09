# -*- coding: utf-8 -*-
import os

import numpy as np
import pytest

import obspy
import obspy.geodetics.base as geodetics
from obspy.taup.ray_paths import get_ray_paths
from obspy.taup import TauPyModel


class TestRayPathCalculations:
    """
    Test suite for obspy.taup.ray_paths
    """
    def setUp(self):
        # load an inventory and an event catalog to test
        # the ray path routines. Careful, the full catalog
        # test is quite long and is therefore commented out
        # by default
        self.path = os.path.join(os.path.dirname(__file__), 'images')
        pass

    @pytest.mark.skipif(
        not geodetics.GEOGRAPHICLIB_VERSION_AT_LEAST_1_34,
        reason='test needs geographiclib >= 1.34')
    def test_compute_ray_paths(self):
        # careful, the full inventory, catalog test is long (1min)
        # greatcircles = get_ray_paths(
        #        inventory=self.inventory, catalog=self.catalog,
        #        phase_list=['P'], coordinate_system='XYZ',
        #        taup_model='iasp91')

        # this test checks if we get a single P wave greatcircle
        station = obspy.core.inventory.Station(
            code='STA', latitude=0., longitude=30., elevation=0.)
        # make two (identical) stations
        network = obspy.core.inventory.Network(
            code='NET', stations=[station, station])
        inventory = obspy.core.inventory.Inventory(
            source='ME', networks=[network])

        otime = obspy.UTCDateTime('2017-02-03T12:00:00.0Z')
        origin = obspy.core.event.Origin(latitude=0., longitude=90.,
                                         depth=100000., time=otime)
        origin.resource_id = 'smi:local/just-a-test2'
        magnitude = obspy.core.event.Magnitude(mag=7.)
        event = obspy.core.event.Event(origins=[origin],
                                       magnitudes=[magnitude])
        event.resource_id = 'smi:local/just-a-test'
        # make three (identical) events
        catalog = obspy.core.event.Catalog(events=[event, event, event])

        # query for four phases
        greatcircles = get_ray_paths(
            inventory, catalog, phase_list=['P', 'PP', 'S', 'SS'],
            coordinate_system='XYZ', taup_model='iasp91')
        # two stations, three events, 4 phases should yield 24 rays (since all
        # arrivals are encountered in this case)
        assert len(greatcircles) == 24
        # now check details of first ray
        circ = greatcircles[0]
        path = circ[0]
        assert circ[1] == 'P'
        assert circ[2] == 'NET.STA'
        np.testing.assert_allclose(circ[3], otime.timestamp, atol=1e-5, rtol=0)
        assert circ[4] == 7.0
        assert circ[5] == 'smi:local/just-a-test'
        assert circ[6] == 'smi:local/just-a-test2'
        assert path.shape == (3, 274)
        # now check some coordinates of the calculated path, start, end and
        # some values in between
        path_start_expected = [
            [6.02712296e-17, 4.63135005e-05, 1.82928142e-03, 1.99453947e-03,
             2.16015881e-03],
            [9.84303877e-01, 9.84224340e-01, 9.81162947e-01, 9.80879369e-01,
             9.80595408e-01],
            [6.02712296e-17, 6.02663595e-17, 6.00790075e-17, 6.00616631e-17,
             6.00442971e-17]]
        np.testing.assert_allclose(path[:, :5], path_start_expected, rtol=1e-5)
        path_end_expected = [
            [8.65195410e-01, 8.65610142e-01, 8.65817499e-01,
             8.66024850e-01, 8.66025746e-01],
            [4.99866617e-01, 4.99932942e-01, 4.99966104e-01,
             4.99999264e-01, 4.99999407e-01],
            [6.11842455e-17, 6.12082668e-17, 6.12202774e-17,
             6.12322880e-17, 6.12323400e-17]]
        np.testing.assert_allclose(path[:, -5:], path_end_expected, rtol=1e-5)
        path_steps_expected = [
            [6.02712296e-17, 5.99694796e-03, 1.55844904e-02,
             2.29391617e-02, 3.12959401e-02, 4.94819381e-02,
             6.59026261e-02, 8.84601669e-02, 1.15734196e-01,
             1.30670566e-01, 1.83202229e-01, 2.21387617e-01,
             3.00609265e-01, 4.51339383e-01, 5.39194024e-01,
             5.84050335e-01, 6.48856399e-01, 6.68018760e-01,
             7.03962600e-01, 7.34809690e-01, 7.59887900e-01,
             7.89619836e-01, 8.04521516e-01, 8.18115511e-01,
             8.36517555e-01, 8.48367905e-01, 8.59911335e-01,
             8.65610142e-01],
            [9.84303877e-01, 9.74082937e-01, 9.58372986e-01,
             9.46927810e-01, 9.34554991e-01, 9.10758513e-01,
             8.91322841e-01, 8.68804460e-01, 8.43147543e-01,
             8.29702355e-01, 7.85418525e-01, 7.55840616e-01,
             7.00517755e-01, 6.14315731e-01, 5.74088249e-01,
             5.56174903e-01, 5.33353699e-01, 5.27297943e-01,
             5.16800783e-01, 5.08777243e-01, 5.04479841e-01,
             5.00872221e-01, 4.99943609e-01, 4.99408317e-01,
             4.99111122e-01, 4.99125255e-01, 4.99180374e-01,
             4.99932942e-01],
            [6.02712296e-17, 5.96465079e-17, 5.86911789e-17,
             5.79996164e-17, 5.72570664e-17, 5.58501220e-17,
             5.47267635e-17, 5.34739746e-17, 5.21120017e-17,
             5.14308206e-17, 4.93839986e-17, 4.82263481e-17,
             4.66770017e-17, 4.66770017e-17, 4.82263481e-17,
             4.93839986e-17, 5.14308206e-17, 5.21120017e-17,
             5.34739746e-17, 5.47267635e-17, 5.58501220e-17,
             5.72570664e-17, 5.79996164e-17, 5.86911789e-17,
             5.96465079e-17, 6.02712296e-17, 6.08831943e-17,
             6.12082668e-17]]
        np.testing.assert_allclose(path[:, ::10], path_steps_expected,
                                   rtol=1e-5)

        # now do the same for RTP coordinate system and also use a different
        # model
        # query for four phases
        greatcircles = get_ray_paths(
            inventory, catalog, phase_list=['P', 'PP', 'S', 'SS'],
            coordinate_system='RTP', taup_model='ak135')
        # two stations, three events, 4 phases should yield 24 rays (since all
        # arrivals are encountered in this case)
        assert len(greatcircles) == 24
        # now check details of first ray
        circ = greatcircles[0]
        path = circ[0]
        assert circ[1] == 'P'
        assert circ[2] == 'NET.STA'
        np.testing.assert_allclose(circ[3], otime.timestamp, atol=1e-5, rtol=0)
        assert circ[4] == 7.0
        assert circ[5] == 'smi:local/just-a-test'
        assert circ[6] == 'smi:local/just-a-test2'
        assert path.shape == (3, 270)
        # now check some coordinates of the calculated path, start, end and
        # some values in between
        path_start_expected = [
            [0.984304, 0.984217, 0.981165, 0.980880, 0.980595],
            [1.570796, 1.570796, 1.570796, 1.570796, 1.570796],
            [1.570796, 1.570745, 1.568935, 1.568765, 1.568595]]
        np.testing.assert_allclose(path[:, :5], path_start_expected, rtol=1e-6)
        path_end_expected = [
            [0.998124, 0.999062, 0.999531, 0.999765, 1.000000],
            [1.570796, 1.570796, 1.570796, 1.570796, 1.570796],
            [0.524316, 0.523957, 0.523778, 0.523688, 0.523599]]
        np.testing.assert_allclose(path[:, -5:], path_end_expected, rtol=1e-6)
        path_steps_expected = [
            [0.98430388, 0.97410140, 0.95847208, 0.94717382, 0.93508421,
             0.91210171, 0.89145501, 0.86719412, 0.84963114, 0.83022016,
             0.80548417, 0.78780343, 0.76158226, 0.76488692, 0.79524407,
             0.80633663, 0.83604439, 0.85093573, 0.87434336, 0.89536778,
             0.91994977, 0.93757572, 0.94908037, 0.95919008, 0.97447903,
             0.98726847, 0.99568357],
            [1.57079633, 1.57079633, 1.57079633, 1.57079633, 1.57079633,
             1.57079633, 1.57079633, 1.57079633, 1.57079633, 1.57079633,
             1.57079633, 1.57079633, 1.57079633, 1.57079633, 1.57079633,
             1.57079633, 1.57079633, 1.57079633, 1.57079633, 1.57079633,
             1.57079633, 1.57079633, 1.57079633, 1.57079633, 1.57079633,
             1.57079633, 1.57079633],
            [1.57079633, 1.56464945, 1.55454336, 1.54659206, 1.53738096,
             1.51661510, 1.49423077, 1.46055656, 1.43236890, 1.39637248,
             1.33996718, 1.28795961, 1.16342477, 0.91617848, 0.79110880,
             0.76040906, 0.69478715, 0.66799188, 0.63152179, 0.60344720,
             0.57851384, 0.56322297, 0.55460834, 0.54755404, 0.53770068,
             0.53004245, 0.52531799]]
        np.testing.assert_allclose(path[:, ::10], path_steps_expected,
                                   rtol=1e-6)

    def test_deep_source(self):
        # Regression test -- check if deep sources are ok
        model = TauPyModel("ak135")
        arrivals = model.get_ray_paths(2000.0, 60.0, ["P"])
        assert abs(arrivals[0].time - 480.32) < 1e-2
