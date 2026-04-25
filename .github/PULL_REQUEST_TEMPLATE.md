<!--
Before submitting a PR, please review the pull request guidelines:
https://github.com/obspy/obspy/blob/master/CONTRIBUTING.md#submitting-a-pull-request
-->

### What does this PR do?

*Please fill in*

### Links to other Issues?

*Please link any relevant issues with "see #123456", mark issues that get resolved with "fixes #12345"*

### AI used?

*If AI tools were used in creating the PR please indicate here how they were used*

### PR Checklist

- [ ] Correct **base branch** selected? [`master` for new features, `maintenance_?.?.x` for bug fixes](https://github.com/obspy/obspy/wiki/ObsPy-Git-Branching-Model)
- [ ] **Tests**: Added new tests for any new features or fixed regressions
- [ ] **Changelog**: Added a short note in `CHANGELOG.txt` (only obsolete if fixing a bug introduced *after* the last release)
- [ ] Add the yellow `ready for review` label when you the PR is **ready to be reviewed**

Rare actions items:

- [ ] First time contributors: Feel free to add your name to `CONTRIBUTORS.txt`
- [ ] New modules: add the module to `CODEOWNERS` with your github handle

### Issue labels

The PR can be flagged with the following "issue labels" to alter CI runs:
- `no_ci` to skip CI builds while work-in-progress
- `build_docs` to trigger automatic docs build to [see how docs render for the PR](https://docs.obspy.org/pr/)
- `test_network` to include "networked" tests if the PR touches any tests marked as "network"
- `upload_images` to attach plots generated in tests as artifacts to the CI run
