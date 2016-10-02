import os
import re
import requests
import warnings
from obspy import UTCDateTime


try:
    # github API token with "repo.status" access right
    token = os.environ["OBSPY_COMMIT_STATUS_TOKEN"]
except KeyError:
    msg = ("Could not get authorization token for ObsPy github API "
           "(env variable OBSPY_COMMIT_STATUS_TOKEN)")
    warnings.warn(msg)
    HEADERS = None
else:
    HEADERS = {"Authorization": "token {}".format(token)}


def check_docs_build_requested(issue_number):
    """
    Check if a docs build was requested for given issue number (by magic string
    '+DOCS' anywhere in issue comments).

    :rtype: bool
    """
    url = "https://api.github.com/repos/obspy/obspy/issues/{:d}/comments"
    data_ = requests.get(url.format(issue_number),
                         params={"per_page": 100}, headers=HEADERS)
    data = data_.json()
    url = data_.links.get("next", {}).get("url", None)
    while url:
        data_ = requests.get(url, headers=HEADERS)
        data += data_.json()
        url = data_.links.get("next", {}).get("url", None)
    if not isinstance(data, list):
        from pprint import pprint
        msg = "Unexpected response from github API:\n{}".format(pprint(data))
        raise Exception(msg)
    comments = [x["body"] for x in data]
    pattern = r'\+DOCS'
    return any(re.search(pattern, comment) for comment in comments)


def get_open_pull_requests():
    """
    Fetch a list of issue numbers for open pull requests (max. 100, no
    pagination), recently updated first, along with the PR data.
    """
    data = requests.get(
        "https://api.github.com/repos/obspy/obspy/pulls",
        params={"state": "open", "sort": "updated", "direction": "desc",
                "per_page": 100},
        headers=HEADERS)
    try:
        assert data.ok
    except:
        print(data.json())
        raise
    data = data.json()
    open_prs = []
    for d in data:
        number = d['number']
        fork = d['head']['user']['login']
        branch = d['head']['ref']
        commit = d['head']['sha']
        open_prs.append((number, fork, branch, commit, d))
    return open_prs


def get_commit_status(commit, context=None):
    """
    Return current commit status. Either for a specific context, or overall.

    :type commit: str
    :type context: str
    :param context: Commit status context (as a str) or ``None`` for overall
        commit status.
    :rtype: str or ``None``
    :returns: Current commit status (overall or for specific context) as a
        string or ``None`` if given context has no status.
    """
    url = "https://api.github.com/repos/obspy/obspy/commits/{}/status".format(
        commit)
    r = requests.get(url, headers=HEADERS)
    try:
        assert r.ok
    except:
        print(r.json())
        raise
    data = r.json()

    if context is None:
        return data['state']

    state = [status['state'] for status in data['statuses']
             if status['context'] == context]
    return state and state[0] or None


def get_commit_time(commit, fork="obspy"):
    """
    """
    url = "https://api.github.com/repos/{fork}/obspy/git/commits/{hash}"
    url = url.format(fork=fork, hash=commit)
    commit_data = requests.get(url, headers=HEADERS)
    try:
        assert commit_data.ok
    except:
        print(commit_data.json())
        raise
    commit_data = commit_data.json()
    return UTCDateTime(commit_data['committer']['date'])


def get_issue_numbers_that_need_docs_build(verbose=False):
    """
    Relies on a local directory with some files to mark when PR docs have been
    built etc.
    """
    open_prs = get_open_pull_requests()
    if verbose:
        print("Checking the following open PRs if a docs build is requested "
              "and needed: {}".format(str(num for num, _ in open_prs)))

    for number, fork, branch, commit, data in open_prs:
        if not check_docs_build_requested(number):
            continue
        # need to figure out time of last push from commit details.. -_-
        time = get_commit_time(commit, fork)
        if verbose:
            print("PR #{} requests a docs build, latest commit {} at "
                  "{}.".format(number, commit, time))
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
                if verbose:
                    print("PR #{} was last built at {} and does not need a "
                          "new build.".format(number, time_done))
                continue
        # ..otherwise touch the .todo file
        with open(filename_todo, "wb"):
            if verbose:
                print("PR #{} build has been queued.".format(number))

    if verbose:
        print("Done checking which PRs require a docs build.")


def set_commit_status(commit, status, context, description,
                      target_url=None, fork="obspy", only_when_changed=True,
                      only_when_no_status_yet=False, verbose=False):
    """
    :param only_when_changed: Whether to only set a status if the commit status
        would change (commit statuses can not be updated or deleted and there
        is a limit of 1000 commit status per commit).
    :param only_when_no_status_yet: Whether to only set a status if the commit
        has no status with given context yet.
    """
    if status not in ("success", "pending", "error", "failure"):
        raise ValueError("Invalid status: {}".format(status))

    # check current status, only set a status if it would change the current
    # status..
    # (avoid e.g. flooding with "pending" status on continuously breaking docs
    #  builds that get started over and over again..)
    # if status would not change.. do nothing, don't send that same status
    # again
    if only_when_changed or only_when_no_status_yet:
        current_status = get_commit_status(commit, context)
        if only_when_no_status_yet:
            if current_status is not None:
                if verbose:
                    print("Commit {} already has a commit status ({}), "
                          "skipping.".format(commit, current_status))
                return
        if only_when_changed:
            if current_status == status:
                if verbose:
                    print("Commit {} status would not change ({}), "
                          "skipping.".format(commit, current_status))
                return

    url = "https://api.github.com/repos/obspy/obspy/statuses/{}".format(commit)
    data = {"state": status, "context": context, "description": description}
    if target_url:
        data["target_url"] = target_url
    r = requests.post(url, json=data, headers=HEADERS)

    try:
        assert r.ok
    except:
        print(r.json())
        raise
    if verbose:
        print("Set commit {} status (context '{}') to '{}'.".format(
            commit, context, status))


def set_all_updated_pull_requests_docker_testbot_pending(verbose=False):
    """
    Set a status "pending" for all open PRs that have not been processed by
    docker buildbot yet.
    """
    open_prs = get_open_pull_requests()
    if verbose:
        print("Working on PRs: " + ", ".join(
            [str(number) for number, _, _, _, _ in open_prs]))
    for number, fork, branch, commit, data in open_prs:
        set_commit_status(
            commit=commit, status="pending", context="docker-testbot",
            description="docker testbot results not available yet",
            only_when_no_status_yet=True,
            verbose=verbose)
