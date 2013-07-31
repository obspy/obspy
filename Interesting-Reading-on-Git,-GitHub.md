 - github development workflow
   - http://scottchacon.com/2011/08/31/github-flow.html
   - http://nvie.com/posts/a-successful-git-branching-model/
 - convert existing issue into a pull request
   - can be done using [defunkt/hub](https://github.com/defunkt/hub)
   - or via a simple POST command, e.g. using `curl`:
```bash
curl --user megies --data '{"issue": "2", "head": "megies:testbranch2", "base": "master"}' https://api.github.com/repos/megies/test/pulls
```
      - **issue**: number of the *already existing* normal issue
      - **head**: *repository/branch* that should be pulled in
      - **base**: branch that the pull request should be merged into (target repository specified in the url)
 - interaction with svn
   - https://github.com/blog/1178-collaborating-on-github-with-subversion