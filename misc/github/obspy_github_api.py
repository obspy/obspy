# -*- coding: utf-8 -*-
import copy
import os
import re
import warnings

import github3

from obspy import UTCDateTime
from obspy.core.util.base import DEFAULT_MODULES, ALL_MODULES


# regex pattern in comments for requesting a docs build
PATTERN_DOCS_BUILD = r'\+DOCS'
# regex pattern in comments for requesting tests of specific submodules
PATTERN_TEST_MODULES = r'\+TESTS:([a-zA-Z0-9_\.,]*)'


try:
    # github API token with "repo.status" access right (if used to set commit
    # statuses) or with empty scope; to get around rate limitations
    token = os.environ["OBSPY_COMMIT_STATUS_TOKEN"]
except KeyError:
    msg = ("Could not get authorization token for ObsPy github API "
           "(env variable OBSPY_COMMIT_STATUS_TOKEN)")
    warnings.warn(msg)
    token = None

gh = github3.login(token=token)


def check_module_tests_requested(issue_number):
    """
    Check if tests of specific modules are requested for given issue number
    (e.g. by magic string '+TESTS:clients.fdsn,clients.arclink' or '+TESTS:ALL'
    anywhere in issue description or comments).
    Accumulates any occurrences of the above magic strings.

    :rtype: list
    :returns: List of modules names to test for given issue number.
    """
    issue = gh.issue("obspy", "obspy", issue_number)
    modules_to_test = set(copy.copy(DEFAULT_MODULES))

    # process issue body/description
    match = re.search(PATTERN_TEST_MODULES, issue.body)
    if match:
        modules = match.group(1)
        if modules == "ALL":
            return ALL_MODULES
        modules_to_test = set.union(modules_to_test, modules.split(","))

    # process issue comments
    for comment in issue.comments():
        match = re.search(PATTERN_TEST_MODULES, comment.body)
        if match:
            modules = match.group(1)
            if modules == "ALL":
                return ALL_MODULES
            modules_to_test = set.union(modules_to_test, modules.split(","))

    return sorted(list(modules_to_test))


def check_docs_build_requested(issue_number):
    """
    Check if a docs build was requested for given issue number (by magic string
    '+DOCS' anywhere in issue comments).

    :rtype: bool
    """
    issue = gh.issue("obspy", "obspy", issue_number)
    if re.search(PATTERN_DOCS_BUILD, issue.body):
        return True
    for comment in issue.comments():
        if re.search(PATTERN_DOCS_BUILD, comment.body):
            return True
    return False


def get_open_pull_requests():
    """
    Fetch a list of issue numbers for open pull requests (max. 100, no
    pagination), recently updated first, along with the PR data.
    """
    repo = gh.repository("obspy", "obspy")
    prs = repo.pull_requests(state="open", sort="updated", direction="desc")
    # (number, fork name, head branch name, head commit SHA)
    open_prs = [(pr.number, pr.head.user.login, pr.head.ref, pr.head.sha)
                for pr in prs]
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
    # github3.py seems to lack support for fetching the "current" statuses for
    # all contexts.. (which is available in "combined status" for an SHA
    # through github API)
    repo = gh.repository("obspy", "obspy")
    commit = repo.commit(commit)
    statuses = {}
    for status in commit.statuses():
        if (status.context not in statuses or
                status.updated_at > statuses[status.context].updated_at):
            statuses[status.context] = status

    # just return current status for given context
    if context:
        if context not in statuses:
            return None
        return statuses[context].state

    # return a combined status
    statuses = set(status.state for status in statuses.values())
    for status in ("pending", "error", "failure", "success"):
        if status in statuses:
            return status

    return None


def get_commit_time(commit, fork="obspy"):
    """
    """
    repo = gh.repository(fork, "obspy")
    commit = repo.commit(commit)
    return UTCDateTime(commit.commit.committer["date"])


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

    repo = gh.repository(fork, "obspy")
    commit = repo.commit(commit)
    repo.create_status(sha=commit.sha, state=status, context=context,
                       description=description, target_url=target_url)
    if verbose:
        print("Set commit {} status (context '{}') to '{}'.".format(
            commit.sha, context, status))


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
