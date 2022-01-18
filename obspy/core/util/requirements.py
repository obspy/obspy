"""
A python module for ObsPy's requirements.

These are kept in this module so they are accessible both from the setup.py
and from the ObsPy namespace.
"""
# Hard dependencies needed to install/run ObsPy.
INSTALL_REQUIRES = [
    'numpy>=1.15.0',
    'scipy>=1.0.0',
    'matplotlib>=3.2.0',
    'lxml',
    'setuptools',
    'sqlalchemy',
    'decorator',
    'requests']

# The modules needed for running ObsPy's test suite.
PYTEST_REQUIRES = [
    'packaging',
    'pytest',
    'pytest-cov',
    'pytest-json-report',
]

# Extra dependencies
EXTRAS_REQUIRES = {
    'tests': ['pyproj'] + PYTEST_REQUIRES,
    # arclink decryption also works with: pycrypto, m2crypto, pycryptodome
    'arclink': ['cryptography'],
    'io.shapefile': ['pyshp'],
    }

# Soft dependencies to include in ObsPy test report.
SOFT_DEPENDENCIES = ['cartopy', 'flake8', 'geographiclib', 'pyproj',
                     'shapefile']
