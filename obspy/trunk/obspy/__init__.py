# -*- coding: utf-8 -*-

"""Python for Seismological Observatories."""

# See http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)


from obspy.core import Trace, Stream, read

#try:
#    import core
#except:
#    pass
#try:
#    import filter
#except:
#    pass
#try:
#    import gse2
#except:
#    pass
#try:
#    import imaging
#except:
#    pass
#try:
#    import mseed
#except:
#    pass
#try:
#    import numpy
#except:
#    pass
#try:
#    import parser
#except:
#    pass
#try:
#    import picker
#except:
#    pass
#try:
#    import sac
#except:
#    pass
#try:
#    import util
#except:
#    pass
#try:
#    import wav
#except:
#    pass
#try:
#    import xseed
#except:
#    pass
