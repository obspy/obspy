# -*- coding: utf-8 -*-

"""Python for Seismological Observatories."""

# A namespace package will not have any method, class or any other
# objects defined in that level.
# @see: http://baijum81.livejournal.com/22412.html

# DON'T IMPORT HERE ANYTHING!

# See http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)
