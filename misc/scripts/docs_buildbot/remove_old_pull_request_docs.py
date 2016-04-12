import glob
import os
import requests
import shutil
from obspy import UTCDateTime


DIRECTORY = "/home/obspy/htdocs/docs/pull_requests"


try:
    # github API token with "repo.status" access right
    token = os.environ["OBSPY_COMMIT_STATUS_TOKEN"]
except KeyError:
    headers = None
else:
    headers = {"Authorization": "token {}".format(token)}


# without using pagination we only get the last 100 pulls,
# but this should be enough
data = requests.get(
    "https://api.github.com/repos/obspy/obspy/pulls",
    params={"state": "closed", "sort": "updated", "direction": "desc",
            "per_page": 100},
    headers=headers)
try:
    assert data.ok
except:
    print(data.json())
    raise

now = UTCDateTime()
# delete everything belonging to pull requests that have been closed for more
# than two weeks
time_threshold = now - 14 * 24 * 3600
# any files older than this will be deleted no matter what
time_threshold_hard = (now - 365 * 24 * 3600).timestamp


def delete(path):
    try:
        if os.path.isfile(file_):
            os.remove(file_)
        elif os.path.isdir(file_):
            shutil.rmtree(file_)
    except Exception as e:
        print("Failed to remove '{}' ({}).".format(file_, str(e)))


for d in data.json():
    # extract the pieces we need from the PR data
    number = d['number']
    time = UTCDateTime(d['closed_at'])

    # still pretty freshly closed, so leave it alone
    if time > time_threshold:
        continue

    for file_ in glob.glob(
            os.path.join(DIRECTORY, str(number) + "*")):
        delete(file_)

for file_ in glob.glob(os.path.join(DIRECTORY, "*")):
    if os.stat(file_).st_mtime < time_threshold_hard:
        delete(file_)
