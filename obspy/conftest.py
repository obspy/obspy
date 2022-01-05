#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Obspy's testing configuration file.
"""
import numpy as np
import pytest

from obspy.core.util import NETWORK_MODULES


@pytest.fixture(scope='session', autouse=True)
def set_numpy_print_options():
    """
    Make sure the doctests print the same style of output across all numpy
    versions.
    """
    try:
        np.set_printoptions(legacy='1.13')
    except (TypeError, AttributeError):
        pass


@pytest.fixture(scope='session', autouse=True)
def set_mpl_backend():
    """
    Make sure the doctests print the same style of output across all numpy
    versions.
    """
    import matplotlib
    matplotlib.use('Agg')


def pytest_addoption(parser):
    parser.addoption('--network', action='store_true', default=False,
                     help='test network modules', )
    # other options
    others = parser.getgroup('Additional Options')
    others.addoption('--tutorial', action='store_true',
                     help='add doctests in tutorial')
    others.addoption('--keep-images', action='store_true',
                     help='store images created during image comparison '
                          'tests in subfolders of baseline images')
    others.addoption('--keep-only-failed-images', action='store_true',
                     help='when storing images created during testing, '
                          'only store failed images and the corresponding '
                          'diff images (but not images that passed the '
                          'corresponding test).')


def pytest_collection_modifyitems(config, items):
    """ Preprocessor for collected tests. """
    network_nodes = set(NETWORK_MODULES)
    for item in items:
        # get the obspy model test originates from (eg clients.arclink)
        obspy_node = '.'.join(item.nodeid.split('/')[1:3])
        # if test is a network test apply network marker
        # TODO apply proper marks
        if obspy_node in network_nodes:
            item.add_marker(pytest.mark.network)


def pytest_runtest_setup(item):
    """Setup test runs."""
    # skip network tests if not specified by command line argument.
    if 'network' in item.keywords and not item.config.getvalue("--network"):
        pytest.skip("need --network option to run")


def pytest_configure(config):
    """
    If the network option is not set skip all network tests
    """
    if not config.getoption('--network'):
        setattr(config.option, 'markexpr', 'not network')
    # set print options
    try:
        np.set_printoptions(legacy='1.13')
    except (TypeError, AttributeError):
        pass
    # ensure matplotlib doesn't print anything to the screen
    import matplotlib
    matplotlib.use('Agg')
