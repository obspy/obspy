# This Makefile requires GNU make, sometimes available as gmake.
#
# A simple test suite for libmseed.
# See README for description.
#
# Build environment can be configured the following
# environment variables:
#   CC : Specify the C compiler to use
#   CFLAGS : Specify compiler options to use

# Required compiler parameters
CFLAGS += -I..

LDFLAGS = -L..
LDLIBS = -lmseed

SRCS := $(wildcard *.c)
BINS := $(SRCS:%.c=%)

TESTS := $(wildcard *.test)
TESTOUTS := $(TESTS:%.test=%.test.out)

# ASCII color coding for test results, green for PASSED and red for FAILED
PASSED := "\033[0;32mPASSED\033[0m"
FAILED := "\033[0;31mFAILED\033[0m"

test all: $(BINS) $(TESTOUTS)
	@echo All tests conducted.

# Build programs and check for executable
$(BINS) : % : %.c
	@$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS); exit 0;
	@if test -x $@; \
	  then echo $(PASSED) "Building $<"; \
	  else echo $(FAILED) "Building $<"; exit 1;\
        fi

# Run test scripts, create %.test.out files and compare to %.test.ref references
$(TESTOUTS) : %.test.out : %.test FORCE
	@$(shell ./$< > $@ 2>&1)
	@diff $<.ref $@ >/dev/null; if [ $$? -eq 0 ]; \
          then echo $(PASSED) "Test $<"; \
          else echo $(FAILED) "Test $<, Compare $<.ref to $@"; exit 0;\
        fi

clean:
	@rm -f $(BINS) $(TESTOUTS)

# Any targets using this empty FORCE rule as a prerequisite will always run
FORCE:
