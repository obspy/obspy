import os
from datetime import datetime

from obspy_github_api import (
    check_docs_build_requested, get_pull_requests, get_commit_time)


print("Checking the open PRs if a docs build is requested and needed..")

for pr in get_pull_requests(state="open"):
    number = pr.number
    fork = pr.head.user.login
    branch = pr.head.ref
    commit = pr.head.sha

    if not check_docs_build_requested(number):
        print("PR #{} does not request a docs build.".format(number))
        continue

    time = get_commit_time(commit=commit, fork=fork)
    print("PR #{} requests a docs build, latest commit {} at {}.".format(
        number, commit, str(datetime.fromtimestamp(time))))

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
        time_done = os.stat(filename_done).st_atime
        if time_done > time:
            print("PR #{} was last built at {} and does not need a "
                  "new build.".format(
                      number, str(datetime.fromtimestamp(time_done))))
            continue
    # ..otherwise touch the .todo file and set it's time to the time of the
    # last commit +1 second (we'll use that time as the time of the docs build)
    with open(filename_todo, "wb"):
        print("PR #{} build has been queued.".format(number))
    os.utime(filename_todo, (time + 1, time + 1))

print("Done checking which PRs require a docs build.")
