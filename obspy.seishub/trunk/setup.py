# -*- coding: utf-8 -*-

"""
setup.py bdist_egg
"""

from setuptools import setup, find_packages

version = '0.0.1'


setup(
    name='obspy.seishub',
    version=version,
    description="",
    long_description="""
    obspy.seishub
    =============
    """,
    classifiers=[],
    keywords='Seismology SeisHub',
    author='Robert Barsch',
    author_email='barsch@lmu.de',
    url='https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.seishub',
    license='GPL',
    packages=find_packages(exclude=['ez_setup']),
    namespace_packages=['obspy'],
    include_package_data=True,
    zip_safe=False,
    #test_suite = "obspy.gse2.tests.suite",
    install_requires=[
        'setuptools',
        'numpy',
        'lxml',
        'matplotlib',
    ],
    download_url="https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.seishub/trunk#egg=obspy.seishub-dev",
)
