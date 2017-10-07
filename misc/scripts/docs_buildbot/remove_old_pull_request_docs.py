import glob
import os
import shutil
import warnings

warnings.filterwarnings(
    "ignore", "Matplotlib is building the font cache", UserWarning)

from obspy import UTCDateTime

from obspy_github_api import get_pull_requests


DIRECTORY = "/home/obspy/htdocs/docs/pull_requests"


prs = get_pull_requests(state="closed", sort="updated", direction="desc")

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


for pr in prs:
    # extract the pieces we need from the PR data
    number = pr.number
    time = UTCDateTime(pr.closed_at)

    # still pretty freshly closed, so leave it alone
    if time > time_threshold:
        continue

    for file_ in glob.glob(
            os.path.join(DIRECTORY, str(number) + "*")):
        delete(file_)

for file_ in glob.glob(os.path.join(DIRECTORY, "*")):
    if os.stat(file_).st_mtime < time_threshold_hard:
        delete(file_)
