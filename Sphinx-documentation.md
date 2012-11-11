The API reference documentation of ObsPy is generated with Sphinx. In order to build it locally by you own, run
```
#!sh
cd /path/to/obspy/trunk/misc/docs
make pep8
make html
```

Sphinx version 1.1 or higher has to be installed (e.g. via ```easy_install sphinx==1.1```). In case of error messages involving ```matplotlib.sphinxext``` try updating the matplotlib installation.

A fully self-contained script to install a local Python environment suitable for building the docs from scratch is available here: https://github.com/obspy/sandbox/blob/master/buildbots/install_python.sh

More useful links on Sphinx:
 * [Sphinx video on showmedo](http://showmedo.com/videotutorials/video?name=2910020&fromSeriesID=291)
 * [Official web page](http://sphinx.pocoo.org)
 * [Sphinx markup fields](http://sphinx.pocoo.org/markup/desc.html?highlight=params#info-field-lists)