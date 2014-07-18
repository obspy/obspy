#include <setjmp.h>
#include "evresp.h"


void _obspy_set_jump() {
    setjmp(jump_buffer);
}
