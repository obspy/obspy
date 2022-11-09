<!--

Thank your for contributing to ObsPy!

!! Please check that you select the **correct base branch** (details see below link) !!

Before submitting a PR, please review the pull request guidelines:
https://github.com/obspy/obspy/blob/master/CONTRIBUTING.md#submitting-a-pull-request

Also, please make sure you are following the ObsPy branching model:
https://github.com/obspy/obspy/wiki/ObsPy-Git-Branching-Model

-->

### What does this PR do?

*Please fill in*

### Why was it initiated?  Any relevant Issues?

*Please fill in (link relevant issues with "see #123456", mark issues that get resolved with "fixes #12345")*

### PR Checklist
- [ ] Correct base branch selected? `master` for new features, `maintenance_...` for bug fixes
- [ ] This PR is not directly related to an existing issue (which has no PR yet).
- [ ] While the PR is still work-in-progress, the `no_ci` label can be added to skip CI builds
- [ ] If the PR is making changes to documentation, docs pages can be built automatically.
      Just add the `build_docs` tag to this PR.
      Docs will be served at [docs.obspy.org/pr/{branch_name}](https://docs.obspy.org/pr/) (do not use master branch).
      Please post a link to the relevant piece of documentation.
- [ ] If all tests including network modules (e.g. `clients.fdsn`) should be tested for the PR,
      just add the `test_network` tag to this PR.
- [ ] All tests still pass.
- [ ] Any new features or fixed regressions are covered via new tests.
- [ ] Any new or changed features are fully documented.
- [ ] Significant changes have been added to `CHANGELOG.txt` .
- [ ] First time contributors have added your name to `CONTRIBUTORS.txt` .
- [ ] If the changes affect any plotting functions you have checked that the plots
      from all the CI builds look correct. Add the "upload_plots" tag so that plotting 
      outputs are attached as artifacts. 
- [ ] New modules, add the module to `CODEOWNERS` with your github handle
- [ ] Add the `ready for review` tag when you are ready for the PR to be reviewed.
