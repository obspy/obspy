
# Build environment can be configured the following
# environment variables:
#   CC : Specify the C compiler to use
#   CFLAGS : Specify compiler options to use

# Required compiler parameters
CFLAGS += -I..

LDFLAGS = -L..
LDLIBS = -lmseed

all: msview msrepack

msview: msview.o
	$(CC) $(CFLAGS) -o $@ msview.o $(LDFLAGS) $(LDLIBS)

msrepack: msrepack.o
	$(CC) $(CFLAGS) -o $@ msrepack.o $(LDFLAGS) $(LDLIBS)

clean:
	rm -f msview.o msview msrepack.o msrepack

install:
	@echo
	@echo "No install target, copy the executable(s) as needed"
	@echo
