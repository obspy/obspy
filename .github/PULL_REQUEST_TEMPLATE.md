<!--
Please check that you select the **correct base branch** (details see below link)

Before submitting a PR, please review the pull request guidelines:
https://github.com/obspy/obspy/blob/master/CONTRIBUTING.md#submitting-a-pull-request

Also, please make sure you are following the ObsPy branching model:
https://github.com/obspy/obspy/wiki/ObsPy-Git-Branching-Model
-->

### What does this PR do?

*Please fill in*

### Links to other Issues?

*Please link any relevant issues with "see #123456", mark issues that get resolved with "fixes #12345"*

### Use of A.I. tools?

*If AI tools were used in creating the PR please indicate here how they were used*

### PR Checklist
- [ ] Correct base branch selected? `master` for new features, `maintenance_...` for bug fixes
- [ ] Tests: Added new tests for any new features or fixed regressions
- [ ] Changelog: Added a short note in `CHANGELOG.txt` (not applicable only if PR affects changes that are not in a released version)
- [ ] First time contributors: Feel free to add your name to `CONTRIBUTORS.txt`
- [ ] New modules: add the module to `CODEOWNERS` with your github handle
- [ ] Add the yellow `ready for review` label when you are ready for the PR to be reviewed.

### Issue labels

The PR can be flagged with the following "issue labels":
- "no_ci": skip CI builds while work-in-progress
- "build_docs": if needed, trigger automatic docs build to [see how docs render for the PR](https://docs.obspy.org/pr/)
- "test_network": if any tests marked as "network" are touched by the PR, add this to run them in CI
- "upload_images": if PR adds/changes any plots, this will attach plot output as artifacts in CI
