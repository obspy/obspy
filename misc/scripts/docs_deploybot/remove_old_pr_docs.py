from datetime import datetime, timezone
from glob import glob
import logging
import os
import shutil


logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('remove_docs')
GLOB = '/home/obspy/htdocs/docs/pr/*'


def remove_old_docs(days=90, globexpr=GLOB):
    now = datetime.now(tz=timezone.utc).timestamp()
    time = now - days * 24 * 3600
    log.info(f'Find PR docs older than {days} days')
    for path in glob(globexpr):
        if os.path.getmtime(path) < time:
            log.info(f'Remove old PR docs at {path}')
            shutil.rmtree(path)


if __name__ == '__main__':
    remove_old_docs()
