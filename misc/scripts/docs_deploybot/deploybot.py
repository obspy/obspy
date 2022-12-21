"""
Hi, I am ObsPy's docs deploy bot. I request github runs and
extract uploaded doc artifacts to the ObsPy server.
Outdated PR docs older than 90 days will be deleted.
"""
import logging
from logging.handlers import TimedRotatingFileHandler
import sched
import signal
import sys

import requests

handlers = [TimedRotatingFileHandler('log.txt', 'D', 30, 5)]
format_ = '%(levelname)s:%(name)s:%(asctime)s %(message)s'
datefmt = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=format_, datefmt=datefmt,
                    handlers=handlers)

from deploy_docs import deploy
from remove_old_pr_docs import remove_old_docs


log = logging.getLogger('docsdeploybot')
log.info(' '.join(__doc__.strip().splitlines()))
T1 = 60
T2 = 24 * 3600


def sdeploy():
    try:
        deploy()
    except requests.exceptions.RequestException as ex:
        log.error(str(ex))
    except Exception:
        log.exception('Unexpected error: continue')
    s.enter(T1, 1, sdeploy)


def sremove():
    remove_old_docs()
    s.enter(T2, 2, sremove)


def signal_term_handler(signal, frame):
    log.info('Received SIGTERM: Bye, bye')
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_term_handler)


s = sched.scheduler()
s.enter(0, 1, sdeploy)
s.enter(0, 2, sremove)
try:
    s.run()
except KeyboardInterrupt:
    log.info('Received KeyboardInterrupt: Bye, bye')
except Exception:
    log.exception('Unexpected error: Bye, bye')
