## Makefile to simplify Octave Forge package maintenance tasks

## Some shell programs
MD5SUM    ?= md5sum
SED       ?= sed
GREP      ?= grep
TAR       ?= tar

## Helper function
TOLOWER   := $(SED) -e 'y/ABCDEFGHIJKLMNOPQRSTUVWXYZ/abcdefghijklmnopqrstuvwxyz/'

## Specific for this package
OCTPKG_DIR  := octave_pkg
SUBFOLDERS  := cov mean prior inf lik util help demo
SOURCES     := util/solve_chol.c \
               $(wildcard util/lbfgsb/*.f) $(wildcard util/lbfgsb/*.cpp) \
               $(wildcard util/lbfgsb/*.h) $(wildcard util/minfunc/mex/*.c)
HG_EXCLUDED := --exclude $(OCTPKG_DIR) \
               --exclude "startup.m" \
               --exclude ".octaverc" \
               --exclude "Makefile" \
               --exclude "util/make.m" \
               --exclude "util/minfunc/mex/mexAll_matlab.m" \
               --exclude "util/minfunc/mex/mexAll_octave.m"

## Create needed files
DESC := $(OCTPKG_DIR)/DESCRIPTION

### Note the use of ':=' (immediate set) and not just '=' (lazy set).
### http://stackoverflow.com/a/448939/1609556
PACKAGE := $(shell $(SED) -n -e 's/^Name: *\(\w\+\)/\1/p' $(DESC) | $(TOLOWER))
VERSION := $(shell $(SED) -n -e 's/^Version: *\(\w\+\)/\1/p' $(DESC) | $(TOLOWER))
DEPENDS := $(shell $(SED) -n -e 's/^Depends[^,]*, \(.*\)/\1/p' $(DESC) | $(SED) 's/ *([^()]*),*/ /g')

## This are the files that will be created for the releases.
TARGET_DIR      := OF
RELEASE_DIR     := $(TARGET_DIR)/$(PACKAGE)-$(VERSION)
RELEASE_TARBALL := $(TARGET_DIR)/$(PACKAGE)-$(VERSION).tar.gz
HTML_DIR        := $(TARGET_DIR)/$(PACKAGE)-html
HTML_TARBALL    := $(TARGET_DIR)/$(PACKAGE)-html.tar.gz

## Octave binaries
# Follow jwe suggestion on not inheriting these vars from
# the enviroment, so they can be set as command line arguemnts
MKOCTFILE := mkoctfile
OCTAVE    := octave --no-gui

## To run without installation
## extract directive from register function
PKG_ADD     := $(shell $(GREP) -sPho '(?<=(//|\#\#) PKG_ADD: ).*' \
                         $(OCTPKG_DIR)/__gpml_package_register__.m)

## Targets that are not filenames.
## https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
.PHONY: help dist html release install all check run clean

## make will display the command before runnning them.  Use @command
## to not display it (makes specially sense for echo).
help:
	@echo "Targets:"
	@echo "   dist             - Create $(RELEASE_TARBALL) for release"
	@echo "   html             - Create $(HTML_TARBALL) for release"
	@echo "   release          - Create both of the above and show md5sums"
	@echo
	@echo "   install          - Install the package in GNU Octave"
	@echo "   all              - Build all oct files"
	@echo "   run              - Run Octave with development in PATH (no install)"
	@echo "   check            - Execute package tests (w/o install)"
	@echo
	@echo "   clean            - Remove releases, html documentation, and oct files"

# dist and html targets are only PHONY/alias targets to the release
# and html tarballs.
dist: $(RELEASE_TARBALL)
html: $(HTML_TARBALL)

# An implicit rule with a recipe to build the tarballs correctly.
%.tar.gz: %
	tar -c -f - --posix -C "$(TARGET_DIR)/" "$(notdir $<)" | gzip -9n > "$@"

# Some packages are distributed outside Octave Forge in non tar.gz format.
%.zip: %
	cd "$(TARGET_DIR)" ; zip -9qr - "$(notdir $<)" > "$(notdir $@)"

# Create the unpacked package.
#
# Notes:
#    * having ".hg/dirstate" as a prerequesite means it is only rebuilt
#      if we are at a different commit.
#    * the variable RM usually defaults to "rm -f"
#    * having this recipe separate from the one that makes the tarball
#      makes it easy to have packages in alternative formats (such as zip)
#    * note that if a commands needs to be ran in a specific directory,
#      the command to "cd" needs to be on the same line.  Each line restores
#      the original working directory.
$(RELEASE_DIR): .hg/dirstate $(OCTPKG_DIR)/index.sh $(OCTPKG_DIR)/news.sh
	@echo "Creating package version $(VERSION) release ..."
	$(RM) -r "$@"
	hg archive --exclude ".hg*" $(HG_EXCLUDED) --type files "$@"
	cp $(OCTPKG_DIR)/DESCRIPTION "$@"
	cp $(OCTPKG_DIR)/CITATION "$@"
	cp $(OCTPKG_DIR)/__gpml_package_register__.m "$@"
	@echo
	@echo "Reorganizing documentation files ..."
	$(OCTPKG_DIR)/doc.sh "$@"                # creates manual & demo
	@echo
	@echo "Creating INDEX file ..."
	$(OCTPKG_DIR)/index.sh "$@" > "$@/INDEX"
	@echo
	@echo "Creating package structure ..."
	mkdir "$@/inst" && mkdir "$@/src"
	@echo
	@echo "Reorganizing program files ..."
	cd "$@" && mv $(SOURCES) "src/" && \
	           mv $(SUBFOLDERS) "inst/" && mv *.m "inst/" && \
	           mv inst/util/minfunc/*.m "inst/util/"
	cp $(OCTPKG_DIR)/Makefile "$@/src"
	@echo
	@echo "Creating NEWS file ..."
	$(OCTPKG_DIR)/news.sh > "$@/NEWS"
	@echo
	@echo "Creating Copyright files ..."
	mv "$@/Copyright" "$@/COPYING"
	mv "$@/inst/util/lbfgsb/LICENSE" "$@/src/COPYING_lbfgsb"
	mv "$@/inst/util/minfunc/License" "$@/src/COPYING_minfunc"
	@echo
	@echo "Cleaning up residual folders ..."
	rm -rf "$@/inst/util/lbfgsb" && rm -rf "$@/inst/util/minfunc"
	chmod -R a+rX,u+w,go-w $@

# install is a prerequesite to the html directory (note that the html
# tarball will use the implicit rule for ".tar.gz" files).
$(HTML_DIR): install
	@echo "Generating HTML documentation. This may take a while ..."
	$(RM) -r "$@"
	$(OCTAVE) --no-window-system --norc --silent \
	  --eval "pkg load generate_html; " \
	  --eval "pkg load $(PACKAGE);" \
	  --eval 'generate_package_html ("${PACKAGE}", "$@", "octave-forge");'
	chmod -R a+rX,u+w,go-w $@

# To make a release, build the distribution and html tarballs.
release: dist html
	@$(MD5SUM) $(RELEASE_TARBALL) $(HTML_TARBALL)
	@echo "Upload @ https://sourceforge.net/p/octave/package-releases/new/"
	@echo "    and inform to rebuild release with '$$(hg id)'"

# We need to avoid calling startup.m
# First option was to move to $(TARGET_DIR) but that is dirty.
# Mike Miller indicated that --no-site-file avoids running the system's
# octaverc file, which contains the code to run startup.m and finish.m
# There seems to be no harm in avoiding that file since it only configures
# readline. Note: the user's .octaverc is still read.
install: $(RELEASE_TARBALL)
	@echo "Installing package locally ..."
	$(OCTAVE) --silent --no-site-file --eval 'pkg ("install", "-verbose", "$(RELEASE_TARBALL)")'

clean:
	$(RM) -r $(RELEASE_DIR) $(RELEASE_TARBALL) $(HTML_TARBALL) $(HTML_DIR)

# Build any required mex/oct file
# Here we do make all because there are several targets
# We do clean at the end to remove *.o files which are not necessary
all: $(CC_SOURCES) $(SOURCES) $(RELEASE_DIR)
	$(MAKE) -C $(RELEASE_DIR)/src/ $@
	$(MAKE) -C $(RELEASE_DIR)/src/ clean

# Start an Octave session with the package directories on the path for
# interactice test of development sources.
run: all
	cd "$(RELEASE_DIR)/inst" && \
	$(OCTAVE) --silent --persist --norc \
	  --eval 'if(!isempty("$(DEPENDS)")); pkg load $(DEPENDS); endif;' \
	  --eval '$(PKG_ADD)' \
	  --eval 'addpath("../src")'

#
# Recipes for testing purposes
#
check: all
