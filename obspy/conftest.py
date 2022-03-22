"""
Obspy's testing configuration file.
"""
import argparse
import os
import platform
import shutil
import sys
from pathlib import Path
from subprocess import run
from contextlib import suppress

import numpy as np
import pytest

import obspy
from obspy.core.util import NETWORK_MODULES


OBSPY_PATH = os.path.dirname(obspy.__file__)


# Soft dependencies to include in ObsPy test report.
REPORT_DEPENDENCIES = ['cartopy', 'flake8', 'geographiclib', 'pyproj',
                       'shapefile', 'pytest', 'pytest-json-report']

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
    if request.config.getoption('--keep-images', default=False):
        new_path = Path(OBSPY_PATH) / 'obspy_test_images'
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
    yield new_path
    # finally close all figs created by this test
    from matplotlib.pyplot import close
    close('all')


# --- Pytest configuration

class ToggleAction(argparse.Action):
    def __call__(self, parser, ns, values, option):
        setattr(ns, self.dest, option[2:4] != 'no')


def pytest_addoption(parser):
    """Pytest hook which allows setting package-specific command-line args."""
    parser.addoption('--network', action='store_true', default=False,
                     help='test only network modules', )
    parser.addoption('--all', action='store_true', default=False,
                     help='run both network and non-network tests', )
    parser.addoption('--report', '--no-report', dest='report',
                     action=ToggleAction, nargs=0,
                     help='Generate a json report of the test results and '
                          'upload it to ObsPys test server.',)
    parser.addoption('--keep-images', action='store_true',
                     help='store images created while runing test suite '
                          'in a directory called obspy_test_images.')


def pytest_collection_modifyitems(config, items):
    """ Preprocessor for collected tests. """
    network_module_names = set(NETWORK_MODULES)
    for item in items:
        # explicitely add filter warnings to markers so that they have a higher
        # priority than command line options, e.g. -W error
        for fwarn in config.getini('filterwarnings'):
            item.add_marker(pytest.mark.filterwarnings(fwarn))
        # automatically apply image mark to tests using image_path fixture.
        if 'image_path' in getattr(item, 'fixturenames', {}):
            item.add_marker('image')
        # Mark network doctests, network test files are already marked.
        name_split = item.name.replace('obspy.', '').split('.')
        if len(name_split) >= 2:
            possible_obspy_node = '.'.join(name_split[:2])
            if possible_obspy_node in network_module_names:
                item.add_marker(pytest.mark.network)


def pytest_configure(config):
    """
    Configure pytest with custom logic for ObsPy before test run.
    """
    # Skip or select network options based on options
    network_selected = config.getoption('--network')
    all_selected = config.getoption('--all')
    if network_selected and not all_selected:
        setattr(config.option, 'markexpr', 'network')
    elif not network_selected and not all_selected:
        setattr(config.option, 'markexpr', 'not network')

    # Set numpy print options to try to not break doctests.
    try:
        np.set_printoptions(legacy='1.13')
    except (TypeError, AttributeError):
        pass

    # Ensure matplotlib doesn't try to show anything.
    import matplotlib
    matplotlib.use('Agg')


@pytest.hookimpl(optionalhook=True)
def pytest_html_report_title(report):
    """
    A pytest-html hook to add custom fields to the report.
    """
    # Customize the title of the html report (if used) to include version.
    report.title = f"Obspy Tests ({obspy.__version__})"


@pytest.hookimpl(optionalhook=True)
def pytest_json_modifyreport(json_report):
    """Modifies the json report after everything has run."""
    # Add architectural info
    json_report['platform_info'] = get_environmental_info()
    # Add github actions info
    json_report['ci_info'] = get_github_actions_info()
    # Add version dependencies
    json_report['dependencies'] = get_dependency_info()
    json_report['runtest_flags'] = ' '.join(sys.argv[1:])
    # Add log for compat. with obspy reporter. We can use this in the
    # future to attach log files if needed.
    json_report['log'] = None
    # Remove any possible environmental variables included in Packages:
    # https://github.com/obspy/obspy/pull/2489#issuecomment-1009806258
    with suppress((KeyError, ValueError)):  # don't let this fail tests
        packages = json_report.get('environment', {})
        filtered_packages = {
            i: v for i, v in packages.items() if not i.isupper()
        }
        json_report['environment'] = filtered_packages


def get_dependency_info():
    """Add version info about ObsPy's dependencies."""
    import pkg_resources
    distribution = pkg_resources.get_distribution('obspy')
    version_info = {'obspy': obspy.__version__}
    for req in distribution.requires():
        name = req.name
        version = pkg_resources.get_distribution(name).version
        version_info[name] = version
    for name in REPORT_DEPENDENCIES:
        try:
            version = pkg_resources.get_distribution(name).version
        except pkg_resources.DistributionNotFound:
            version = '---'
        version_info[name] = version
    return version_info


def get_github_actions_info():
    """
    Adds information from github actions environmental variables.
    """
    ci_info = {}
    # Some of these are added to the CI by obspy's setup.
    vars = ['ISSUE_NUMBER', 'PR_URL', 'CI_URL',
            'RUNNER_OS', 'RUNNER_ARCH', 'GITHUB_JOB', 'GITHUB_WORKFLOW',
            'GITHUB_ACTION', 'GITHUB_SHA', 'GITHUB_EVENT_NAME',
            "GITHUB_ACTOR"]
    for name in vars:
        save_name = name.lower()
        ci_info[save_name] = os.environ.get(name)
    return ci_info


def get_environmental_info():
    """Add info to dict about platform/architecture."""
    # get system / environment settings
    platform_info = {}
    for name in ['system', 'release', 'version', 'machine',
                 'processor', 'python_version', 'python_implementation',
                 'python_compiler', 'architecture']:
        try:
            temp = getattr(platform, name)()
            if isinstance(temp, tuple):
                temp = temp[0]
            platform_info[name] = temp
        except Exception:
            platform_info[name] = ''
    # add node name
    node_name = os.environ.get('OBSPY_NODE_NAME')
    if not node_name:
        node_name = platform.node().split('.', 1)[0]
    platform_info['node'] = node_name
    return platform_info
