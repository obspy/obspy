# -*- coding: utf-8 -*-
"""
The obspy.imaging.backend test suite.
"""
import matplotlib


class TestBackend:
    """
    Test cases for matplotlib backend.

    Note: This test will fail when called from an interactive Python session
    where matplotlib was already imported.
    """
    def test_backend(self):
        """
        Test to see if tests are running without any X11 or any other display
        variable set. Therefore, the AGG backend is chosen in
        obspy.imaging.tests.__init__, and nothing must be imported before,
        e.g. by obspy.imaging.__init__. The AGG backend does not require and
        display setting. It is therefore the optimal for programs on servers
        etc.
        """
        assert 'AGG' == matplotlib.get_backend().upper()
