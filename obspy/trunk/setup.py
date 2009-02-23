"""
setup.py bdist_egg
"""

from setuptools import setup, find_packages

version = '0.0.1'

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
    namespace_packages=['obspy'],
    include_package_data=True,
    zip_safe=False,
    # test_suite = "obspy.gse2.tests",
    install_requires=[
        'setuptools',
    ],
)
