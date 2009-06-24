"""
setup.py bdist_egg
"""

from setuptools import setup, find_packages

version = '0.0.3'

setup(
    name='obspy',
    version=version,
    description="",
    long_description="""""",
    classifiers=[],
    keywords='Seismology',
    author='Moritz Beyreuther',
    author_email='beyreuth@geophysik.uni-muenchen.de',
    url='https://svn.geophysik.uni-muenchen.de/svn/obspy',
    license='GPL',
    packages=find_packages(exclude=['ez_setup']),
    namespace_packages=[],
    include_package_data=True,
    zip_safe=True,
    # test_suite = "obspy.gse2.tests",
    install_requires=[
        'setuptools',
        'numpy',
        'matplotlib',
        'obspy.core',
        'obspy.gse2',
        'obspy.mseed',
        'obspy.imaging'
    ],
    download_url="https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy/trunk#egg=obspy-dev",
    dependency_links=[
        "https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.core/trunk#egg=obspy.core-dev",
        "https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.gse2/trunk#egg=obspy.gse2-dev",
        "https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.mseed/trunk#egg=obspy.mseed-dev",
        "https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.imaging/trunk#egg=obspy.imaging-dev",
    ],
)
