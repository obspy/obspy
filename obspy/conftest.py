"""
Obspy's testing configuration file.
"""
import shutil
from pathlib import Path
from subprocess import run

import numpy as np
import pytest

import obspy
from obspy.core.util import NETWORK_MODULES


# --- ObsPy fixtures

@pytest.fixture(scope='class')
def ignore_numpy_errors():
    """
    Ignore numpy errors for marked tests.
    """
    nperr = np.geterr()
    np.seterr(all='ignore')
    yield
    np.seterr(**nperr)


@pytest.fixture(scope='session', autouse=True)
def save_image_directory(request, tmp_path_factory):
    """
    Creates a temporary directory for storing all images.
    """
    tmp_image_path = tmp_path_factory.mktemp('images')
    yield Path(tmp_image_path)
    # if keep images is selected then we move images to directory
    # and add info about environment.
    if request.config.getoption('--keep-images'):
        new_path = Path().cwd() / 'obspy_test_images'
        if new_path.exists():  # get rid of old image folder
            shutil.rmtree(new_path)
        shutil.copytree(tmp_image_path, new_path)
        # todo probably need to handle this in a more OS robust way
        # but since it is mainly for CI running with it for now.
        run(f"python -m pip freeze > {new_path/'pip_freeze.txt'}", shell=True)
        run(f"conda list > {new_path / 'conda_list.txt'}", shell=True)


@pytest.fixture(scope='function')
def image_path(request, save_image_directory):
    """
    Returns a path for saving an image.

    These will be saved to obspy_test_images if --keep-images is selected.
    Using this fixture will also mark a test with "image".
    """
    parent_obj = getattr(request.node.parent, 'obj', None)
    node_name = request.node.name
    if parent_obj:  # add parent class to name, parent.name doesn't work
        if hasattr(parent_obj, '__name__'):
            parent_name = parent_obj.__name__
        else:
            parent_name = str(parent_obj.__class__.__name__)
        node_name = parent_name + '_' + node_name
    new_path = save_image_directory / (node_name + '.png')
    return new_path


# --- Pytest configuration


def pytest_addoption(parser):
    """Pytest hook which allows setting package-specfic command-line args."""
    parser.addoption('--network', action='store_true', default=False,
                     help='test only network modules', )
    parser.addoption('--all', action='store_true', default=False,
                     help='run both network and non-network tests', )
    parser.addoption('--coverage', action='store_true', default=False,
                     help='Report Obspy Coverage to terminal and generate '
                          'xml report which will be saved as coverage.xml', )
    parser.addoption('--report', action='store_true', default=False,
                     help='Generate and HTML report of test results '
                          'saved as obspy_report.html in obspy directory.', )
    parser.addoption('--keep-images', action='store_true',
                     help='store images created while runing test suite '
                          'in a directory called obspy_test_images.')


def pytest_collection_modifyitems(config, items):
    """ Preprocessor for collected tests. """
    network_nodes = set(NETWORK_MODULES)

    for item in items:
        # get the obspy model test originates from (eg clients.arclink)
        obspy_node = '.'.join(item.nodeid.split('/')[1:3])
        # if test is a network test apply network marker
        # We need to keep these to properly mark doctests, event though
        # the test files now have proper marks.
        if obspy_node in network_nodes:
            item.add_marker(pytest.mark.network)

        # automatically apply image mark to tests using image_path fixture.
        if 'image_path' in getattr(item, 'fixturenames', {}):
            item.add_marker('image')


def pytest_configure(config):
    """
    Configure pytest with custom logic for ObsPy before test run.
    """
    # Add doctest option so all doctests run
    config.option.doctestmodules = True

    # Skip or select network options based on options
    network_selected = config.getoption('--network')
    all_selected = config.getoption('--all')
    if network_selected and not all_selected:
        setattr(config.option, 'markexpr', 'network')
    elif not network_selected and not all_selected:
        setattr(config.option, 'markexpr', 'not network')

    # select appropriate options for report
    # this is the same as --html=obspy_report.html --self-contained-html
    if config.getoption('--report'):
        config.option.htmlpath = 'obspy_report.html'
        config.option.self_contained_html = True

    # select options for coverage
    # this is the same as using:
    # --cov obspy --cov-report term-missing --cov-report='xml' --cov-append
    if config.getoption('--coverage'):
        config.option.cov_report = {'term-missing': None, 'xml': None}
        config.known_args_namespace.cov_source = ['obspy']
        config.option.cov_append = True
        # this is a bit hinky, but we need to register and load coverage here
        # or else it doesn't work
        from pytest_cov.plugin import CovPlugin
        config.option.cov_source = ['obspy']
        options = config.known_args_namespace
        plugin = CovPlugin(options, config.pluginmanager)
        config.pluginmanager.register(plugin, '_cov')

    # Set numpy print options to try to not break doctests.
    try:
        np.set_printoptions(legacy='1.13')
    except (TypeError, AttributeError):
        pass

    # Ensure matplotlib doesn't try to show anything.
    import matplotlib
    matplotlib.use('Agg')

    # Register markers. We should really do this in pytest.ini or the like
    # but the existence of that file messed with the config.py hooks for
    # some reason.
    config.addinivalue_line(
        "markers", "network: Test requires network resources (internet)."
    )
    config.addinivalue_line(
        "markers", "image: Test produces a matplotlib image."
    )


def pytest_html_report_title(report):
    """
    A pytest-html hook to add custom fields to the report.
    """
    # Customize the title of the html report (if used) to include version.
    report.title = f"Obspy Tests ({obspy.__version__})"
