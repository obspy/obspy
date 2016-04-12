import os
import requests
from obspy import UTCDateTime


try:
    # github API token with "repo.status" access right
    token = os.environ["OBSPY_COMMIT_STATUS_TOKEN"]
except KeyError:
    headers = None
else:
    headers = {"Authorization": "token {}".format(token)}


data = requests.get(
    "https://api.github.com/repos/obspy/obspy/pulls",
    params={"state": "open", "sort": "created", "direction": "desc",
            "per_page": 100},
    headers=headers)
try:
    assert data.ok
except:
    print(data.json())
    raise
data = data.json()

for d in data:
    # extract the pieces we need from the PR data
    number = d['number']
    fork = d['head']['user']['login']
    branch = d['head']['ref']
    commit = d['head']['sha']
    # need to figure out time of last push from commit details.. -_-
    url = "https://api.github.com/repos/{fork}/obspy/git/commits/{hash}"
    url = url.format(fork=fork, hash=commit)
    commit_data = requests.get(url, headers=headers)
    try:
        assert commit_data.ok
    except:
        print(commit_data.json())
        raise
    commit_data = commit_data.json()
    time = int(UTCDateTime(commit_data['committer']['date']).timestamp)

    filename = os.path.join("pull_request_docs", str(number))
    filename_todo = filename + ".todo"
    filename_done = filename + ".done"

    # create new stub file if it doesn't exist
    if not os.path.exists(filename):
        with open(filename, "wb") as fh:
            fh.write("{}\n{}\n".format(fork, branch).encode("UTF-8"))

    # update access/modify time of file
    os.utime(filename, (time, time))

    # check if nothing needs to be done..
    if os.path.exists(filename_done):
        time_done = UTCDateTime(os.stat(filename_done).st_atime)
        if time_done > time:
            continue
    # ..otherwise touch the .todo file
    with open(filename_todo, "wb"):
        pass
