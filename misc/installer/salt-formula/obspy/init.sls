#
# Salt - ObsPy install
# Only supported obspy install methods are used (deb + pip)
#
# Use the native package management system for dependencies
# whenever possible. Use pip for everything else.
#
#######################################################################
# REPOSITORY INSTALL
#######################################################################

#- Debian/Ubuntu
#
{% if grains['os_family'] == 'Debian' %}

obspy:
    pkgrepo.managed:
        - humanname: ObsPy - Python framework for seismology
        - name: deb http://deb.obspy.org {{ grains['oscodename'] }} main
        - file: /etc/apt/sources.list.d/obspy.list
        - key_url: https://raw.github.com/obspy/obspy/master/misc/debian/public.key

    pkg.installed:
        - name: python-obspy
        - refresh: True

#######################################################################
# PIP INSTALL
#######################################################################
{% else %}

include:
    - obspy.pip

{% endif %}
