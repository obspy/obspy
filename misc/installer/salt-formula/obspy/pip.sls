#
# Salt state for a pip install of ObsPy (from PyPI)
#
include:
    - obspy.dependencies

obspy:
    pip.installed:
        {% if grains['os_family'] == 'Arch' %}
        - bin_env: /usr/sbin/pip2
        {% endif %}
        - require:
            - pkg: obspy_pkg_dependencies

