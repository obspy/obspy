"""
Jinja templates for HTML reprs.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import jinja2

autoescape = jinja2.select_autoescape(
    default=True,
    default_for_string=True
)

repr_templates_env = jinja2.Environment(
    loader=jinja2.PackageLoader(
        package_name='obspy.html_templates',
        package_path='repr'
    ),
    autoescape=autoescape
)
