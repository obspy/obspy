Thank your for submitting a pull request to ObsPy to help develop and improve
it.

If this is a new feature, send a pull request into the `master` branch. Make
sure you initially branched from the `master` branch. Otherwise please rebase
on top of it. If this is a bugfix, send a PR into the latest
`maintenance_1.x.x` branch. Again make sure you originally branched from that
branch, otherwise rebase. This cannot be changed after the pull request has
been submitted so please make sure to get it correct the first time. See our
[git branching
model](https://github.com/obspy/obspy/wiki/ObsPy-Git-Branching-Model) and our
[general contribution
guide](https://github.com/obspy/obspy/blob/master/CONTRIBUTING.md) for more
details.

All the submitted pieces including potential data must be compatible with the
LGLPv3 license and will be LGPLv3 licensed as soon as they are part of ObsPy.
Sending a pull request implies that you agree with this.

Additionally take care to not add big files. Even for tests we generally only
accept files that are very small and on the order of a few kilobytes.

Before this can be merged, the following requirements must also be fulfilled.
Note that these can also be fulfilled after the pull request has been opened.

- [ ] All tests still pass.
- [ ] Any new features or fixed regressions must be covered via new tests.
- [ ] Any new or changed features have to be fully documented.
- [ ] If the change is significant enough to warrant it, add it to the [Changelog](https://github.com/obspy/obspy/blob/master/CHANGELOG.txt).
- [ ] If this is your first time contributing, please add your name to the [Contributors List](https://github.com/obspy/obspy/blob/master/obspy/CONTRIBUTORS.txt).
