#
# Example for obspy dev, MUST have all dependecies installed already
#

# This forces an upgrade, which must be done with no dep installs, b/c
# otherwise pip will try and build a new numpy/scipy/matplotlib, etc.
# which is probably not what we want.

obspy:
    pip.installed:
        - name : git+https://github.com/obspy/obspy.git@master#egg=obspy
        - activate: True
        - upgrade: True
        - no_deps: True
        - bin_env: /opt/virtualenv/testpy

