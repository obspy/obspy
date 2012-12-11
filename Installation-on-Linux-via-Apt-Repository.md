For installation add the ObsPy apt repository to your package manager by adding the following line to the file `/etc/apt/sources.list` (needs root privileges):
ObsPy is packed for all Debian and Ubuntu releases which are still officially supported.
Replace **CODENAME** with the  codename of your [Debian](http://en.wikipedia.org/wiki/Debian#Releases)/[Ubuntu](http://en.wikipedia.org/wiki/Ubuntu_releases) release.

```sources.list
deb http://deb.obspy.org CODENAME main
```

Currently supported releases are:

 * Debian:
   - squeeze (i386, amd64)
   - wheezy (i386, amd64)
 * Ubuntu:
   - lucid (i386, amd64)
   - natty (i386, amd64)
   - oneiric (i386, amd64)
   - precise (i386, amd64)
   - quantal (i386, amd64)

If you are unsure about the codename of your installation you can use the following command to display it:

```bash
lsb_release -cs
```

Next you need to import the ObsPy gnupg key so that apt is able to check the integrity of the downloaded packages. The following command needs to be executed only once:

```bash
wget --quiet -O - https://raw.github.com/obspy/obspy/master/misc/debian/public.key | sudo apt-key add -
```

To install ObsPy including all dependencies:

```bash
sudo apt-get update
sudo apt-get install python-obspy
```

If you want to uninstall ObsPy and all unneeded dependencies just do:

```bash
sudo apt-get remove python-obspy
```