# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@rm -rf source/packages/autogen/obspy.*

.PHONY: help Makefile

install-obspy-dev:
	python -m pip install ../../

# If you want to build docs from docker where you mounted new version of obspy you have to first reinstall it
html-docker: install-obspy-dev
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

docset: SHELL:=/bin/bash
docset:
	$(eval DOCSETVERSION=`python -c "import obspy; print(obspy.__version__)"`)
	# this should equally work to get the version number
	#$(eval DOCSETVERSION := $(shell head -3 build/html/objects.inv | grep 'Version' | awk '{print $NF}'))
	cd build && doc2dash --verbose --name "ObsPy $(DOCSETVERSION)" --icon html/_static/obspy_logo_no_text_32x32.png --online-redirect-url https://docs.obspy.org/ html/
	sed 's#\(<string>[Oo]bs[Pp]y\) .*\(</string>\)#\1 $(DOCSETVERSION)\2#' --in-place "build/ObsPy $(DOCSETVERSION).docset/Contents/Info.plist"
	cat docset_css_fixes.css >> "build/ObsPy $(DOCSETVERSION).docset/Contents/Resources/Documents/_static/css/custom.css"
	if [[ "$(DOCSETVERSION)" == *"post"* ]]; then \
	  cd build && tar --exclude='.DS_Store' -cvzf "obspy-master.docset.tgz" "ObsPy $(DOCSETVERSION).docset"; \
	else \
	  cd build && tar --exclude='.DS_Store' -cvzf "obspy-$(DOCSETVERSION).docset.tgz" "ObsPy $(DOCSETVERSION).docset"; \
	fi
