#
#
#
include:
    - obspy.dependencies

obspy:
    pip.installed:
        - name : git+https://github.com/obspy/obspy.git@master#egg=obspy
        - require:
            - pkg: obspy_pkg_dependencies

