# -*- coding: utf-8 -*-
"""
USAGE: make_assets.py [-f] [-c]
"""
import os
import shutil
import sys
from urllib import request


CDN_URL = 'https://netdna.bootstrapcdn.com/bootstrap/3.1.1/'

ASSETS = {
    'source/_static/css/base.css': 'http://tests.obspy.org/static/base.css',

    'source/_static/font.css':
        'http://tests.obspy.org/static/font/style.css',
    'source/_static/fonts/icomoon.eot':
        'http://tests.obspy.org/static/font/fonts/icomoon.eot',
    'source/_static/fonts/icomoon.svg':
        'http://tests.obspy.org/static/font/fonts/icomoon.svg',
    'source/_static/fonts/icomoon.ttf':
        'http://tests.obspy.org/static/font/fonts/icomoon.ttf',
    'source/_static/fonts/icomoon.woff':
        'http://tests.obspy.org/static/font/fonts/icomoon.woff',

    'source/_templates/navbar-local.html':
        'http://tests.obspy.org/snippets/navbar.html',
    'source/_templates/footer.html':
        'http://tests.obspy.org/snippets/footer.html',

    'source/_static/css/bootstrap.min.css':
        CDN_URL + 'css/bootstrap.min.css',
    'source/_static/fonts/glyphicons-halflings-regular.eot':
        CDN_URL + 'fonts/glyphicons-halflings-regular.eot',
    'source/_static/fonts/glyphicons-halflings-regular.svg':
        CDN_URL + 'fonts/glyphicons-halflings-regular.svg',
    'source/_static/fonts/glyphicons-halflings-regular.ttf':
        CDN_URL + 'fonts/glyphicons-halflings-regular.ttf',
    'source/_static/fonts/glyphicons-halflings-regular.woff':
        CDN_URL + 'fonts/glyphicons-halflings-regular.woff',
}

force = '-f' in sys.argv
clean = '-c' in sys.argv

if clean:
    print('Cleaning assets ...')
elif force:
    print('Forced downloading assets ...')
else:
    print('Downloading necessary assets ...')

for asset, url in ASSETS.items():
    if clean:
        try:
            print('Deleting %s ...' % (asset))
            os.remove(asset)
        except Exception:
            if force:
                pass
            else:
                raise

    elif force or not os.path.exists(asset):
        print('Downloading %s ...' % (url))
        resp = request.urlopen(url)
        with open(asset, 'wb') as output:
            shutil.copyfileobj(resp, output)
        resp.close()
