# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = build
PYTHON        = python

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

.PHONY: help clean html dirhtml singlehtml pickle json htmlhelp qthelp devhelp \
        epub latex latex-png-images latexpdf-png-images latexpdf text man \
        changes linkcheck doctest gettext coverage coverager pep8 citations \
        credits assets

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  serverhtml to make HTML files for serving online"
	@echo "  pickle     to make pickle files"
	@echo "  json       to make JSON files"
	@echo "  htmlhelp   to make HTML files and a HTML help project"
	@echo "  qthelp     to make HTML files and a qthelp project"
	@echo "  devhelp    to make HTML files and a Devhelp project"
	@echo "  epub       to make an epub"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  text       to make text files"
	@echo "  man        to make manual pages"
	@echo "  texinfo    to make Texinfo files"
	@echo "  info       to make Texinfo files and run them through makeinfo"
	@echo "  gettext    to make PO message catalogs"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation (if enabled)"
	@echo "  coverage   to make create coverage HTML files"
	@echo "  coverager  to make create coverage HTML files and report test"
	@echo "  pep8       to check PEP8 of tutorial"
	@echo "  citations  to generate citations page"
	@echo "  credits    to generate credits page"
	@echo "  assets     to generate static assets for local browsing"

clean:
	@rm -rf $(BUILDDIR)/*
	@rm -rf source/packages/autogen/obspy.*
	@rm -rf source/pep8/*
	@$(PYTHON) make_assets.py -c -f

coverage:
	@coverage run --rcfile=.coveragerc -m obspy.scripts.runtests -q --all
	@coverage html --rcfile=.coveragerc -d $(BUILDDIR)/html/coverage
	@coverage xml --rcfile=.coveragerc -o $(BUILDDIR)/html/coverage.xml

coverager:
	@coverage run --rcfile=.coveragerc -m obspy.scripts.runtests -r -q --all
	@coverage html --rcfile=.coveragerc -d $(BUILDDIR)/html/coverage
	@coverage xml --rcfile=.coveragerc -o $(BUILDDIR)/html/coverage.xml

c_coverage:
	@$(PYTHON) make_c_coverage.py $(BUILDDIR)/html/c_coverage

pep8:
	@$(PYTHON) make_pep8.py 1
	@pep8 --show-source source/tutorial; echo

citations:
	@$(PYTHON) make_citations.py

credits:
	@$(PYTHON) make_credits.py

assets:
	@$(PYTHON) make_assets.py

html: assets credits citations
	if [ ! -d $(BUILDDIR) ]; then mkdir $(BUILDDIR); fi
	if [ ! -d $(BUILDDIR)/html ]; then mkdir $(BUILDDIR)/html; fi
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

serverhtml:
	if [ ! -d $(BUILDDIR) ]; then mkdir $(BUILDDIR); fi
	if [ ! -d $(BUILDDIR)/html ]; then mkdir $(BUILDDIR)/html; fi
	$(SPHINXBUILD) -b html -A use_local_assets=0 $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml:
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

singlehtml:
	$(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(BUILDDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML page is in $(BUILDDIR)/singlehtml."

pickle:
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp:
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp:
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/obspy.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/obspy.qhc"

devhelp:
	$(SPHINXBUILD) -b devhelp $(ALLSPHINXOPTS) $(BUILDDIR)/devhelp
	@echo
	@echo "Build finished."
	@echo "To view the help file:"
	@echo "# mkdir -p $$HOME/.local/share/devhelp/obspy"
	@echo "# ln -s $(BUILDDIR)/devhelp $$HOME/.local/share/devhelp/obspy"
	@echo "# devhelp"

epub:
	$(SPHINXBUILD) -b epub $(ALLSPHINXOPTS) $(BUILDDIR)/epub
	@echo
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

latex-png-images: latex
	bash $(BUILDDIR)/../tex_replace_pdf_suffix.sh $(BUILDDIR)/latex

latexpdf: latex
	@echo "Running LaTeX files through pdflatex..."
	make -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

latexpdf-png-images: latex-png-images latexpdf

text:
	$(SPHINXBUILD) -b text $(ALLSPHINXOPTS) $(BUILDDIR)/text
	@echo
	@echo "Build finished. The text files are in $(BUILDDIR)/text."

man:
	$(SPHINXBUILD) -b man $(ALLSPHINXOPTS) $(BUILDDIR)/man
	@echo
	@echo "Build finished. The manual pages are in $(BUILDDIR)/man."

texinfo:
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) $(BUILDDIR)/texinfo
	@echo
	@echo "Build finished. The Texinfo files are in $(BUILDDIR)/texinfo."
	@echo "Run \`make' in that directory to run these through makeinfo" \
	      "(use \`make info' here to do that automatically)."

info:
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) $(BUILDDIR)/texinfo
	@echo "Running Texinfo files through makeinfo..."
	make -C $(BUILDDIR)/texinfo info
	@echo "makeinfo finished; the Info files are in $(BUILDDIR)/texinfo."

gettext:
	$(SPHINXBUILD) -b gettext $(I18NSPHINXOPTS) $(BUILDDIR)/locale
	@echo
	@echo "Build finished. The message catalogs are in $(BUILDDIR)/locale."

changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

docset: html docset_after_html

docset_after_html:
ifeq ($(DOCSETVERSION),)
	$(eval DOCSETVERSION := $(shell head -3 $(BUILDDIR)/html/objects.inv | grep ' ObsPy Documentation ' | sed 's#.*(\(.*\)).*#\1#'))
endif
	cd $(BUILDDIR)/html && rm -rf icon.png && ln -s _static/obspy-logo_32x32.png icon.png
	rm -rf "$(BUILDDIR)/ObsPy $(DOCSETVERSION).docset"
	cd $(BUILDDIR) && doc2dash -n 'ObsPy $(DOCSETVERSION)' -i html/icon.png html/
	sed 's#\(<string>[Oo]bs[Pp]y\) .*\(</string>\)#\1 ${DOCSETVERSION}\2#' --in-place "$(BUILDDIR)/ObsPy $(DOCSETVERSION).docset/Contents/Info.plist"
	rm -f $(BUILDDIR)/html/icon.png
