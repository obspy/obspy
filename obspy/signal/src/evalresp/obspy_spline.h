 /* Copyright (C) ObsPy Development Team, 2014.
  *
  * This file is licensed under the terms of the GNU Lesser General Public
  * License, Version 3 (http://www.gnu.org/copyleft/lesser.html).
  *
  */

char *evr_spline(int num_points, double *t, double *y, double tension,
                                 double k, double *xvals_arr, int num_xvals,
                                 double **p_retvals_arr, int *p_num_retvals);
