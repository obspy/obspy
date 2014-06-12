/*--------------------------------------------------------------------
# Filename: hermite_interpolation.c
#  Purpose: Performs an hermite interpolation given values and slopes for
#           each point.
#   Author: Lion Krischer
# Copyright (C) 2014 L. Krischer
#---------------------------------------------------------------------*/

/* Converts an array of 32 bit IBM floating point numbers to IEEE
 * floating point numbers.
 *
 * Parameters:
 *	x_in:
 *	y_in:
 *
 * Output will be written to y_out.
 */

 void hermite_interpolation(double *x_in, double *y_in, double *slope,
                            double *x_out, double *y_out,
                            int len_in, int len_out, double h,
                            double x_start) {
    int i_0, i_1;
    double i, t, a_0, a_1, b_minus_1, b_plus_1, b_0, c_0, c_1, d_0;

    for (int idx=0; idx < len_out; idx++) {
        i = (x_out[idx] - x_start) / h;
        i_0 = (int)i;
        i_1 = i_0 + 1;

        // No need to interpolate if exactly at the points.
        if (i == (double)i_0)  {
            y_out[idx] = y_in[i_0];
        }
        if (i == (double)i_1)  {
            y_out[idx] = y_in[i_1];
        }

        t = i - (double)i_0;

        a_0 = y_in[i_0];
        a_1 = y_in[i_1];
        b_minus_1 = h * slope[i_0];
        b_plus_1 = h * slope[i_1];
        b_0 = a_1 - a_0;
        c_0 = b_0 - b_minus_1;
        c_1 = b_plus_1 - b_0;
        d_0 = c_1 - c_0;

        y_out[idx] = a_0 + (b_0 + (c_0 + d_0 * t) * (t - 1.0)) * t;
    }
    return;
}
