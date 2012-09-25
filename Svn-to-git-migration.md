This page contains a short summary of how we performed the migration from svn to git for future reference and identify potential problem down the road.

## 1. Convert the svn repository to git with the help of svn2git
* Get it [here](https://github.com/nirvdrum/svn2git)
* Create a proper authors.txt file to map the svn commiters to git users. github will correctly identify the authors, if the email addresses specified in the file are the same as the ones used on github. See the svn2git homepage for the exact format of the authors.txt file.
* We wanted the new git repository to only contain the current trunk of the repository, but this proved troublesome.
    * The main reason for this is that ObsPy branches mostly are no true branches but rather a collection of scripts, programs, and other things that are not directly related to the actual ObsPy library.
    * In theory, svn2git is able to create git branches and tags from the conventional svn trunk/tags/branches structure. Unfortunately the ObsPy repository did not have proper branches and the whole trunk/tags/branches structure did not exists from the beginning but rather was created at some point in the repository's history.
    * This delayed creation of a proper trunk/tags/branches structure is the reason that the first attempt to solely convert the trunk to git via
  `$ svn2git https://svn.obspy.org --trunk trunk --nobranches --notags --authors ../authors.txt
`
failed because a large part of the history of files was missing because they were originally located somewhere else.
* Therefore the solution was to convert everything to git with with `svn2git https://svn.obspy.org --rootistrunk --authors ../authors.txt` and remove unwanted parts later on.

## 2. Rewrite the repository's history to remove unwanted parts.
* Motivation: The whole repository was around 600 MB in size after the conversion to git. This is largely due to branches, tags, and large binary files checked in at some point in the history of ObsPy but no more present in the latest revision. The size is not a big problem while using svn but become a huge issue with git as every clone will contain the whole history.
* The first step was to remove the tags, branches and trunk/apps directories with `git filter-branch --index-filter "git rm -rf --cached --ignore-unmatch $DIR" HEAD` with `$DIR` being the directory to remove. This command will rewrite the repository's history and simply omit all files in the specified directories.
* The removed files will still reside inside the repository as references and backup. There are some ways to remove those to actually bring down the repository size. This simplest one I found was to create a new empty directory and pull in the one with the rewritten history
```bash
cd ..  
mkdir new_repo
cd new_repo
git init
git pull file:///full/path/to/old/repo
git status  # otherwise git reports an unclean working directory on further commands
```
* This resulted in a repository size of 150MB. Still quite a lot but more manageable.
* To further bring it down we wanted to remove all files that have, at one point in time, been part of the repository but are no longer in the latest revision. This might potentially mess with some more important parts of the history but we felt it is worth it. It proved to be more of a challenge as we wanted to also keep the history of files which had been moved or renamed at some point in time. The final result is short bash/sh script that extracts the necessary information from the repository, then uses a Python script to track file renames/moved to find all files that have no more current version and then once again rewrite the repository's history, excluding the determined files.  
**sh script**
```sh
# Get all the renames in the right order. From old to new.
git log -M --diff-filter=R --name-status --reverse > renamed_files.txt
# Get a list of all files that where ever deleted. Never delete .py files...
git log --diff-filter=D --summary | grep delete | awk '{print $4}' | grep -v .py$ > deleted_files.txt
# The parsing is rather complex, so use Python for it. Will create the
# to_delete.txt function.
python parse_rename_stats.py
# Now permanently remove those files from the repository.
# XXX: The path to the to_delete function is hardcoded! It will be created by
# the above python script.
git filter-branch --prune-empty --index-filter \
    'cat /.../to_delete.txt | xargs git rm -rf --cached --ignore-unmatch' \
    --tag-name-filter cat -- --all

```
**parse_rename_stats.py**
```python
import os
import re

file_tracker = {}

pattern = re.compile("^R[0-9]{1,3}")
with open("./renamed_files.txt", "r") as open_file:
    for line in open_file:
        line = line.strip()
        if re.match(pattern, line) is None:
            continue
        old_file, new_file = line.split("\t")[1:]
        if old_file == new_file:
            continue
        # Loop over the dictionary and get all values.
        available_files = [_i[-1] for _i in file_tracker.values()]
        if not old_file in available_files:
            file_tracker[old_file] = [new_file]
            continue
        # Otherwise search for the correct
        appendeded_stuff = False
        for file_chain in file_tracker.values():
            if file_chain[-1] == old_file:
                file_chain.append(new_file)
                appended_stuff = True
                continue
        if not appended_stuff:
            print "PROBLEM"

# Now check if the file actually still exists on the filesystem. Those will
# keep all history.
files_to_keep = []
for key, value in file_tracker.iteritems():
    if not os.path.exists(value[-1]):
        continue
    files_to_keep.append(key)
    files_to_keep.extend(value)

files_to_permanently_purge = []
# Read all files that have ever been deleted in the repository.
with open("./deleted_files.txt", "r") as open_file:
    for line in open_file:
        line = line.strip()
        if line in files_to_keep:
            continue
        files_to_permanently_purge.append(line)

with open("to_delete.txt", "w") as open_file:
    open_file.write("\n".join(files_to_permanently_purge))
```
* This did work quite well. The history was kept for most files. Only 3 binary files were missing after everything had completed (I am not sure how this could have happened). Those were simply added back into the repository. There still might be some other things missing somewhere in the history of the repository but so far nothing has been found and the state of the latest revision seems fine.
* The final size of the repository is now **67 MB** which should be rather manageable for most people.