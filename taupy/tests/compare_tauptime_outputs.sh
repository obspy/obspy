#!/bin/bash
# Python can't call this as subprocess for some reason
diff -wB <(sort data/java_tauptime_testoutput) <(sort data/taup_time_test_output)