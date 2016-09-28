import os
import re
import requests

from obspy import UTCDateTime

from obspy_github_api import check_docs_build_requested, get_pull_requests, get_commit_time


prs = get_pull_requests(state="open")
pr_numbers = [x[0] for x in prs]
print("Checking the following open PRs if a docs build is requested and "
      "needed: {}".format(str(pr_numbers)))

for pr in prs:
    number = pr.number
    fork = pr.head.user.login
    branch = pr.head.ref
    commit = pr.head.sha

    if not check_docs_build_requested(number):
        continue

    time = get_commit_time(commit=commit, fork=fork)
    print("PR #{} requests a docs build, latest commit {} at {}.".format(
        number, commit, time))
    time = int(time.timestamp)

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
