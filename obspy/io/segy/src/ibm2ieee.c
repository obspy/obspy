/*--------------------------------------------------------------------
# Filename: ibm2ieee.c
#  Purpose: Converts an array of 32 bit IBM floats to IEEE floats.
#   Author: Lion Krischer
# Copyright (C) 2011 L. Krischer
#---------------------------------------------------------------------*/
#include <math.h>
#include <stdlib.h>
#include <stdio.h>


/* Converts an array of 32 bit IBM floating point numbers to IEEE
 * floating point numbers.
 *
 * Parameters:
 *	ibm: Array of 32 bit IBM floating point numbers.
 *	len: Number of samples in the array.
 *
 * It works inplace and thus will not return anything.
 */

void ibm2ieee(float *ibm, int len) {
    int i = 0;
    for (i=0; i<len; i++) {
	/* This very condensed statement was created in an attempt to make it
	 * work with OpenMP but that still seems to suffer from some problems.
	 * See the comment below for a cleaner version.
	 */
	ibm[i] = (((( (*(int*)&ibm[i]) >> 31) & 0x01) * (-2)) + 1) *
	         ((float)((*(int*)&ibm[i]) & 0x00ffffff) / 0x1000000) *
		 ((float)pow(16, (( (*(int*)&ibm[i]) >> 24) & 0x7f) - 64));
    }
    return;
}

/* Easier to read version of the above code. */

//void ibm2ieee(float *ibm, int len) {
//    int sign = 0;
//    int exponent = 0;
//    int mantissa_int = 0;
//    float mantissa_float = 0.0;
//    int ibm_int;
//    int i = 0;
//    for (i=0; i<len; i++) {
//	// Interpret the float as an integer for easier bitshifting.
//	ibm_int = *(int*)&ibm[i];
//	// Get the components of the floating point number.
//	sign = ((( ibm_int >> 31) & 0x01) * (-2)) + 1;
//	exponent = ( ibm_int >> 24) & 0x7f;
//	mantissa_int = ibm_int & 0x00ffffff;
//	double power = (float)pow(16, exponent - 64);
//	mantissa_float = sign * ((float)mantissa_int / 0x1000000) * power;
//	ibm[i] = mantissa_float;
//    }
//    return;
//}
