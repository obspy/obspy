Thank your for submitting a pull request (PR) to ObsPy to help develop and improve it!

First of all: If this PR is directly related to an already existing issue (which is no PR yet), drop us a note in that issue before opening this PR. We can make existing issues into a PR, which avoids duplicated tickets.

Otherwise, if this is a ..

 - **new feature**: make sure that..
   - you choose the `master` branch as the "base" branch of this PR, and...
   - you started your local new branch from the `master` branch (if not rebase your local branch)
 - **bug fix**: make sure that..
   - you choose the latest `maintenance_x.x.x` branch (e.g. `maintenance_1.0.x` if current stable version is `1.0.0` or `1.0.1` etc.) as the "base" branch of this PR, and...
   - you started your local new branch from the latest `maintenance_x.x.x` branch (if not rebase your local branch)

The base branch of the PR cannot be changed after the pull request has been opened so please make sure to get the base branch (`master` vs. `maintenance_x.x.x`) correct the first time. See our [git branching model](https://github.com/obspy/obspy/wiki/ObsPy-Git-Branching-Model) and our [general contribution guide](https://github.com/obspy/obspy/blob/master/CONTRIBUTING.md) for more details.

All the submitted pieces including potential data must be compatible with the LGPLv3 license and will be LGPLv3 licensed as soon as they are part of ObsPy. Sending a pull request implies that you agree with this.

Additionally take care to not add big files. Even for tests we generally only accept files that are very small and at max on the order of a few kilobytes. When in doubt.. ask us in the PR.

Before this can be merged, the following requirements must also be fulfilled. Note that these can also be met after the pull request has been opened, adding more commits to the branch:

- [ ] All tests still pass.
- [ ] Any new features or fixed regressions must be covered via new tests.
- [ ] Any new or changed features have to be fully documented.
- [ ] If the change is significant enough to warrant it, add it to the [Changelog](https://github.com/obspy/obspy/blob/master/CHANGELOG.txt).
- [ ] If this is your first time contributing, please add your name to the [Contributors List](https://github.com/obspy/obspy/blob/master/obspy/CONTRIBUTORS.txt).
