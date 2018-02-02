Thank your for contributing to ObsPy!

!! Please check that you select the **correct base branch** (details see below link) !!

Before submitting a PR, please review the pull request guidelines:
https://github.com/obspy/obspy/blob/master/CONTRIBUTING.md#submitting-a-pull-request

Also, please make sure you are following the ObsPy branching model:
https://github.com/obspy/obspy/wiki/ObsPy-Git-Branching-Model

### What does this PR do?

### Why was it initiated?  Any relevant Issues?

### PR Checklist
- [ ] Correct base branch selected? `master` for new fetures, `maintenance_...` for bug fixes
- [ ] This PR is not directly related to an existing issue (which has no PR yet).
- [ ] If the PR is making changes to documentation, docs pages can be built automatically.
      Just remove the space in the following string after the + sign: "+ DOCS"
- [ ] If any network modules should be tested for the PR, add them as a comma separated list
      (e.g. `clients.fdsn,clients.arclink`) after the colon in the following magic string: "+TESTS:"
      (you can also add "ALL" to just simply run all tests across all modules)
- [ ] All tests still pass.
- [ ] Any new features or fixed regressions are be covered via new tests.
- [ ] Any new or changed features have are fully documented.
- [ ] Significant changes have been added to `CHANGELOG.txt` .
- [ ] First time contributors have added your name to `CONTRIBUTORS.txt` .
