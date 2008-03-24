from distutils.core import setup, Extension

module1 = Extension('ext_pk_mbaer.so',
	include_dirs = ['/usr/lib/python2.4/site-packages/numpy/core/include','/usr/include/python2.4']
	sources = ['pk_mbaer.c','pk_mbaer_wrap.c'])

setup (name = 'BaerPicker',
	version = '0.1',
	author = 'Moritz Beyreuther',
	author_email = 'moritz.beyreuther@geophysik.uni-muenchen.de',
	description = 'C extension for Baer picker',
	ext_modules = [module1])
