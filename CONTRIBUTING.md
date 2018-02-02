# Contributing to ObsPy

This document aims to give an overview of how to contribute to ObsPy. It tries
to answer commonly asked questions, and to provide some insight into how the
community process works in practice.

* If you plan on submitting an issue, please follow this [template](https://github.com/obspy/obspy/blob/master/.github/ISSUE_TEMPLATE.md).
* If you want to open a pull request make sure to read the information on this page but also have a look at our [pull request template](https://github.com/obspy/obspy/blob/master/.github/PULL_REQUEST_TEMPLATE.md).

## Getting Started

 * Make sure you have a GitHub account
 * [Download](https://git-scm.com/downloads) and install git
 * Read the [git documentation](https://git-scm.com/book/en/Git-Basics)
 * Install a [development version of ObsPy](https://github.com/obspy/obspy/wiki/Developer-Installation)

## Submitting a Pull Request

We love pull requests! Here's a quick guide:

First, if the PR is directly related to an already existing issue (which is no PR yet), drop us a note in that issue before opening the PR. We can make existing issues into a PR, which avoids duplicated tickets. Otherwise, please follow the ObsPy [branching model](https://github.com/obspy/obspy/wiki/ObsPy-Git-Branching-Model).

 1. Fork the repo.
 2. Make a new branch. For feature additions/changes base your new branch at `master`, for pure bugfixes base your new branch at e.g. `maintenance_1.0.x` .
 3. Add a test for your change. Only refactoring and documentation changes require no new tests. If you are adding functionality or fixing a bug, we need a test!
 4. Make the test pass.
 5. Push to your fork and submit a pull request.
    - for feature branches set base branch to `obspy:master`
    - for bugfix branches set base branch to the latests maintenance branch, e.g. `obspy:maintenance_1.0.x`
 6. Wait for our review. We may suggest some changes or improvements or alternatives. Keep in mind that PR checklist items can be met after the pull request has been opened by adding more commits to the branch.

Please make sure to select the correct base branch (`master` vs. `maintenance_x.x.x`) for your PR.

If your PR is adding a new submodule, please go through the [to-do items for new submodules in the wiki](https://github.com/obspy/obspy/wiki/How-to%3A-add-a-new-submodule).

If you want to request an automated build of ObsPy's API docs for your PR, simply include the magic string `+DOCS` somewhere in the PR's comments. Once finished, the docs build will show up in the PR's review/commit status section alongside the results from Continuous Integration (PR docs builds can be looked up [here](http://docs.obspy.org/pull-requests/)).

If any specific networking modules should be tested for the PR, e.g. when proposing changes to the FDSN client module, please include a magic string like e.g. `+TESTS:clients.fdsn` somewhere in the PR's comments. Multiple modules can be comma separated (e.g. `+TESTS:clients.fdsn,clients.iris`) or put in separate `+TESTS:...` strings, which will be accumulated across all of the PR's comments. To run *all* modules' test suites, include `+TESTS:ALL` somewhere in a comment.

All the submitted pieces including potential data must be compatible with the LGPLv3 license and will be LGPLv3 licensed as soon as they are part of ObsPy. Sending a pull request implies that you agree with this.

Additionally take care to not add big files. Even for tests we generally only accept files that are very small and at max on the order of a few kilobytes. When in doubt.. ask us in the PR.

## Submitting an Issue

If you want to ask a question about a specific ObsPy aspect, please first of all..

 * search through [Github issues tagged as "question"](https://github.com/obspy/obspy/issues?q=label%3Aquestion)
   (you can add more search terms in the search box, e.g.
   [search for Mini SEED related questions with additional search term "mseed"](
    https://github.com/obspy/obspy/issues?utf8=%E2%9C%93&q=label%3Aquestion+mseed))
 * search the [obspy-users] mailing list archives, e.g.
   [searching for term "mseed" via Google with search string "mseed site:http://lists.swapbytes.de/archives/obspy-users/"](
    https://www.google.de/#q=mseed+site:http:%2F%2Flists.swapbytes.de%2Farchives%2Fobspy-users%2F)

If you want to post a problem/bug, to help us understand and resolve your issue
please check that you have provided the information below:

*  ObsPy version, Python version and Platform (Windows, OSX, Linux ...)
*  How did you install ObsPy and Python (pip, anaconda, from source ...)
*  If possible please supply a [Short, Self Contained, Correct, Example](http://sscce.org/)
      that demonstrates the issue i.e a small piece of code which reproduces
      the issue and can be run with out any other (or as few as possible)
      external dependencies.
*  If this is a regression (Used to work in an earlier version of ObsPy),
      please note when it used to work.

You can also do a quick check whether..

 * the bug was already fixed in the current maintenance branch for the next bug
   fix release, e.g. for the `1.0.x` maintenance line check the..

   * [corresponding changelog section "1.0.x:" right at the top](
      https://github.com/obspy/obspy/blob/maintenance_1.0.x/CHANGELOG.txt)
      which contains bug fixes that will be in the next release) or..

 * if it was already reported and/or is maybe even being worked on already by
   checking open issues of the corresponding milestone, e.g. see

   * [all open issues in milestone "1.0.x"](
      https://github.com/obspy/obspy/milestones/1.0.x) or optionally see..
   * [open and closed issues with "bug" label in that milestone](
      https://github.com/obspy/obspy/issues?utf8=%E2%9C%93&q=milestone%3A1.0.x+label%3Abug+)).

## Additional Resources

 * [Style Guide](https://docs.obspy.org/coding_style.html)
 * [Docs or it doesn't exist!](http://lukeplant.me.uk/blog/posts/docs-or-it-doesnt-exist/)
 * Performance Tips:
    * [Python](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
    * [NumPy and ctypes](https://www.scipy.org/Cookbook/Ctypes)
    * [SciPy](https://wiki.scipy.org/PerformancePython)
    * [NumPy Book](http://csc.ucdavis.edu/~chaos/courses/nlp/Software/NumPyBook.pdf)
 * [Interesting reading on git, github](https://github.com/obspy/obspy/wiki/Interesting-Reading-on-Git%2C-GitHub)
