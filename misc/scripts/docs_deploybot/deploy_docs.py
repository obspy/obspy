# Copyright 2022-2024, GNU LGPL, Obspy developers
from datetime import datetime, timezone
from io import BytesIO
import logging
import os
import re
import shutil
import tarfile
from zipfile import ZipFile

import requests


class DeployError(Exception):
    pass


logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('deploy_docs')
TOKEN = os.getenv('GITHUB_TOKEN', '')
RKW = {'headers': {'accept': 'application/vnd.github.v3+json',
                   'authorization': f'token {TOKEN}'}}
API = 'https://api.github.com/repos/obspy/obspy/'
BASE = '/home/obspy/htdocs/'
PATH = {'pr': BASE + 'docs/pr/{pr}/',
        'master': BASE + 'docs/master/',
        'stable': BASE + 'docs/stable/',  # just a symlink to latest version in archive
        'archive': BASE + 'docs/archive/{v}/'}
URL = 'https://docs.obspy.org/pr/{pr}'

PATH_DOCSET = {'pr': BASE + 'docs/pr/{pr}/docsets/',
               'master': BASE + 'docsets/',
               'stable': BASE + 'docsets/',
               }

PATH_XML = {'pr': BASE + 'docs/pr/{pr}/docsets/obspy-pr.xml',
            'master': BASE + 'docsets/obspy-master.xml',
            'stable': BASE + 'docsets/obspy-stable.xml',
               }

XML = """<entry>
    <version>{version}</version>
    <url>http://docsets.obspy.org/{basename}</url>
</entry>
"""



def get_runs(event=None, time=0):
    data = {'conclusion': 'success', 'per_page': 100}
    r = requests.get(API + 'actions/runs', params=data, **RKW)
    r.raise_for_status()
    runs = r.json()['workflow_runs']
    log.info(f'Requested {len(runs)} runs')
    for run in runs:
        t = datetime.fromisoformat(run['updated_at'][:-1])
        run['time'] = t.replace(tzinfo=timezone.utc).timestamp()
    runs = [run for run in runs if
            run['time'] > time and
            run['name'] == 'docs' and
            run['conclusion'] == 'success' and
            (event is None or run['event'] == event)
            ]
    runs = sorted(runs, key=lambda run: -run['time'])
    log.info(f'Selected {len(runs)} runs')
    return runs


def deploy_artifact(run, other_path=None, other_docset_path=None,
                    overwrite=False):
    if run['event'] == 'pull_request':
        # if len(run['pull_requests']) != 1:
        #     raise DeployError(f"found {len(run['pull_requests'])} PRs")
        # pr = run['pull_requests'][0]['number']
        pr = run['head_branch']
        if pr == 'master':
            raise DeployError('Rename branch')
        path = (other_path or PATH['pr']).format(pr=pr)
        pathd = (other_docset_path or PATH_DOCSET['pr']).format(pr=pr)
        pathxml = PATH_XML['pr'].format(pr=pr)
        overwrite = False
    elif run['event'] == 'push':
        pr = None
        path = (other_path or PATH['master'])
        pathd = (other_docset_path or PATH_DOCSET['master'])
        pathxml = PATH_XML['master']
    elif run['event'] == 'release':
        pr = None
        version = run['head_branch']
        path = (other_path or PATH['archive']).format(v=version)
        pathd = (other_docset_path or PATH_DOCSET['stable']).format(v=version)
        pathxml = PATH_XML['stable']
    else:
        raise DeployError(f"unexpected event {run['event']}")
    msg = (f"Check run {run['id']} triggered by {run['event']}, "
           f"PR: {pr}, {run['conclusion']}")
    log.info(msg)
    if (os.path.exists(path) and
            run['time'] < os.path.getmtime(path) + 1 and
            not overwrite):
        log.info('Run already processed')
        return None, ''
    assert run['conclusion'] == 'success'
    log.info(f'Deploy obspydoc to {path}')
    r = requests.get(run['artifacts_url'], **RKW)
    arts = r.json()['artifacts']
    if len(arts) not in (1, 2):
        raise DeployError(f'Found {len(arts)} artifact')
    log.info('Download obspydoc artifact')
    url = arts[0]['archive_download_url']
    r = requests.get(url, **RKW)
    # the artifact is a zipped tar file
    with ZipFile(BytesIO(r.content)) as z1:
        with z1.open('obspydoc.tar.xz') as z2:
            z2_data = BytesIO(z2.read())
    if os.path.exists(path):
        if run['event'] == 'release' and not overwrite:
            raise DeployError(f'Docs for release {version} already exist')
        log.info('Remove old docs')
        shutil.rmtree(path)
    with tarfile.open(fileobj=z2_data) as tar:
        log.info('Extract obspydoc artifact')
        tar.extractall(path)
    if run['event'] == 'release' and other_path is None:
        log.info('Update obspydoc symlink')
        os.remove(PATH['stable'].rstrip('/'))
        os.symlink(path.rstrip('/'), PATH['stable'].rstrip('/'), True)
    os.utime(path, (run['time'], run['time']))
    log.info('Done with obspydoc')
    if len(arts) == 1:
        return 'success', 'Deployed only doc'

    log.info(f'Deploy obspydocset to {pathd}')
    log.info('Download obspydocset artifact')
    url = arts[1]['archive_download_url']
    r = requests.get(url, **RKW)
    # the artifact is a zipped tar file
    with ZipFile(BytesIO(r.content)) as z1:
        docsettarname = z1.namelist()[0]
        z1.extractall(pathd)  # will overwrite old tgz file
    log.info('Done with obspydocset')

    #  <title>ObsPy Documentation (1.4.1.post0+207.gfef5878495.obspy.master)
    # &mdash; ObsPy 1.4.1.post0+207.gfef5878495.obspy.master documentation</title>
    regex = '<title>.*ObsPy\s*([^\s]*)\s*documentation<\/title>'
    with open(path + 'index.html') as f:
        match = re.search(regex, f.read())
    xml = XML.format(basename=docsettarname, version=match.group(1))
    with open(pathxml, 'w') as f:
        f.write(xml)
    log.info('Done with obspydocset xml')

    return 'success', 'Deployed doc and docset'


def post_state(run, state, msg='Nothing to tell'):
    log.info(f'Post {state} state')
    data = {'state': state,
            'context': f"docs / deploy ({run['event']})",
            'description': msg}
    if run['event'] == 'pull_request':
        try:
            pr = run['head_branch']
        except IndexError:
            pass
        else:
            data['target_url'] = URL.format(pr=pr)
            data['description'] = data['description'] + ', see details link'
    requests.post(API + 'statuses/' + run['head_sha'], json=data, **RKW)


def time2epoch(time):
    if time.lower() == 'none':
        time = 0
    else:
        try:
            time = int(time)
        except ValueError:
            time = datetime.fromisoformat(time).timestamp()
        else:
            now = datetime.now(tz=timezone.utc).timestamp()
            time = now - time
    return time


def deploy(event=None, time='600', **kw):
    runs = get_runs(event=event, time=time2epoch(time))
    for run in runs:
        try:
            state, msg = deploy_artifact(run, **kw)
        except DeployError as ex:
            log.error(f"run {run['id']}: " + str(ex))
            post_state(run, 'failure', msg=str(ex))
        except Exception as ex:
            log.exception(f"Unexpected error in run {run['id']}")
            post_state(run, 'failure', msg=str(ex))
        else:
            if state:
                post_state(run, state, msg)


def run(args=None):
    import argparse
    msg = 'Deploy artifacts from Github Actions on server'
    p = argparse.ArgumentParser(description=msg)
    p.add_argument('--event', choices=['pull_request', 'push', 'release'])
    msg = 'overwrite default path, use only together with --event'
    p.add_argument('--other-path', help=msg)
    msg = 'overwrite default docset path, use only together with --event'
    p.add_argument('--other-docset-path', help=msg)
    msg = 'only for master and latest docs'
    p.add_argument('--overwrite', help=msg, action='store_true')
    msg = ('can be iso-format (UTC) or int (seconds before now), '
           'only use runs after that time (default: 600)')
    p.add_argument('--time', help=msg, default='600')
    args = p.parse_args(args)
    deploy(**vars(args))


if __name__ == '__main__':
    run()
