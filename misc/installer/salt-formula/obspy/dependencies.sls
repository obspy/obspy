#
# Salt state for dependecies available in package repos
#
{% from "obspy/map.jinja" import pkg_deps with context %}

obspy_pkg_dependencies:
    pkg.installed:
        - pkgs:
        {% for pkg_dep in pkg_deps %}
            - {{ pkg_dep }}
        {% endfor %}

