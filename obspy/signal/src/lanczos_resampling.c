/*--------------------------------------------------------------------
# Filename: lanczos_resampling.c
#  Purpose: Lanczos resampling with different kernels
#           each point.
#   Author: Lion Krischer
# Copyright (C) 2015 Lion Krischer and Martin van Driel
#---------------------------------------------------------------------*/

/* Windows apparently needs this */
#define _USE_MATH_DEFINES

#include <math.h>


enum lanczos_window_type {
    LANCZOS = 0,
    HANNING = 1,
    BLACKMAN = 2,
};


/* Sinc function */
static double sinc(double x) {
    if (fabs(x) < 1E-10) {
        return 1.0;
    }
    return sin(M_PI * x) / (M_PI * x);
}


/* Standard Lanczos Kernel */
static double lanczos_kernel(double x, int a) {
    return sinc(x / (double)a);
}


/* von Hann window or raised cosine window*/
static double hanning_kernel(double x, int a) {
    return 0.5 * (1.0 + cos(x / (double)a * M_PI));
}


/* Blackman window
 *
 * Values are taken from http://mathworld.wolfram.com/BlackmanFunction.html
 */
static double blackman_kernel(double x, int a) {
    return 21.0 / 50.0 +
           0.5 * cos(x / (double)a * M_PI) +
           2.0 / 25.0 * cos(2.0 * x / (double)a * M_PI);
}


/* Lanczos resampling with different kernels.
 *
 * Parameters:
 *
 *     y_in: The data values to be interpolated.
 *     y_out: The output array. Must already be initialized with zeros.
 *     dt: The sampling rate factor.
 *     offset: The offset of the first sample in the output array relative
 *         to the input array.
 *     len_in: The length of the input array.
 *     len_out: The length of the output array.
 *     a: The width of the taper in samples on either side.
 *     lanczos_window_type: Which taper window to choose.
 *
 * Output will be written to y_out.
 */
void lanczos_resample(double *y_in, double *y_out, double dt, double offset,
                      int len_in, int len_out, int a,
                      enum lanczos_window_type window) {

    int idx, i, m;
    double x, _x;

    for (idx=0; idx < len_out; idx++) {
        x = dt * idx + offset;
        for (m=-a; m<=a; m++) {
            i = (int)floor(x) - m;
            if (i < 0 || i >= len_in) {
                continue;
            }
            _x = x - i;
            /* Apply kernel and sum up if within 'a' samples on either side. */
            if (-a <= _x && _x <= a) {
                if (window == LANCZOS) {
                    y_out[idx] += y_in[i] * sinc(_x) * lanczos_kernel(_x, a);
                }
                else if (window == HANNING) {
                    y_out[idx] += y_in[i] * sinc(_x) * hanning_kernel(_x, a);
                }
                else if (window == BLACKMAN) {
                    y_out[idx] += y_in[i] * sinc(_x) * blackman_kernel(_x, a);
                }
            }
        }
    }
    return;
}


/* Helper function to be able to analyze and plot the actually used kernels.
 *
 * Parameters:
 *
 *     x: The input x values.
 *     y: The output array. Must have the same number of samples as x.
 *     len: The length of x and y.
 *     a: The width of the taper in samples on either side.
 *     return type: The type of kernel to return. Useful for plotting.
 *         0: Returns the sinc function tapered with the chosen window.
 *         1: Returns only the sinc function.
 *         2: Returns only the chosen taper window.
 *     lanczos_window_type: Which taper window to choose.
 *
 */
void calculate_kernel(double *x, double *y, int len, int a,
                      int return_type, enum lanczos_window_type window) {
    int idx;
    double value;

    for (idx=0; idx<len; idx++) {
        value = x[idx];

        /* Sinc times window */
        if (return_type == 0) {
            if (-a <= value && value <= a) {
                if (window == LANCZOS) {
                    y[idx] = sinc(value) * lanczos_kernel(value, a);
                }
                else if (window == HANNING) {
                    y[idx] = sinc(value) * hanning_kernel(value, a);
                }
                else if (window == BLACKMAN) {
                    y[idx] = sinc(value) * blackman_kernel(value, a);
                }
            }
            else {
                y[idx] = 0.0;
            }
        }
        /* Only sinc */
        else if (return_type == 1) {
            y[idx] = sinc(value);
        }
        /* Only window */
        else if (return_type == 2) {
            if (-a <= value && value <= a) {
                if (window == LANCZOS) {
                    y[idx] = lanczos_kernel(value, a);
                }
                else if (window == HANNING) {
                    y[idx] = hanning_kernel(value, a);
                }
                else if (window == BLACKMAN) {
                    y[idx] = blackman_kernel(value, a);
                }
            }
            else {
                y[idx] = 0.0;
            }
        }

    }
    return;
}
