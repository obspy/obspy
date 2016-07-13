import os
import re
import requests
from obspy import UTCDateTime


def check_docs_build_requested(issue_number, headers=None):
    """
    Check if a docs build was requested for given issue number (by magic string
    '+DOCS' anywhere in issue comments).

    :rtype: bool
    """
    url = "https://api.github.com/repos/obspy/obspy/issues/{:d}/comments"
    data_ = requests.get(url.format(issue_number),
                         params={"per_page": 100}, headers=headers)
    data = data_.json()
    url = data_.links.get("next", {}).get("url", None)
    while url:
        data_ = requests.get(url, headers=headers)
        data += data_.json()
        url = data_.links.get("next", {}).get("url", None)
    if not isinstance(data, list):
        from pprint import pprint
        msg = "Unexpected response from github API:\n{}".format(pprint(data))
        raise Exception(msg)
    comments = [x["body"] for x in data]
    pattern = r'\+DOCS'
    return any(re.search(pattern, comment) for comment in comments)


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

pr_numbers = [d['number'] for d in data]
print("Checking the following open PRs if a docs build is requested and "
      "needed: {}".format(str(pr_numbers)))

for d in data:
    # extract the pieces we need from the PR data
    number = d['number']
    if not check_docs_build_requested(number, headers=headers):
        continue
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
    time = commit_data['committer']['date']
    print("PR #{} requests a docs build, latest commit {} at {}.".format(
        number, commit, time))
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
            print("PR #{} was last built at {} and does not need a "
                  "new build.".format(number, time_done))
            continue
    # ..otherwise touch the .todo file
    with open(filename_todo, "wb"):
        print("PR #{} build has been queued.".format(number))

print("Done checking which PRs require a docs build.")
