/*
   evr_spline():  Cubic spline interpolation.  Two "source" arrays (abscissa
                  'X' and ordinate 'Y' values) and an array of "new"
                  abscissa values are given; an array of "destination"
                  ordinate values is generated and returned.  Any "new"
                  abscissa values outside of the range of "source" abscissa
                  values are ignored, resulting in a reduced-size
                  "destination" ordinate array being generated and an
                  error message being generated.
     num_points:  number of points in given "source" arrays.
     t:  abscissa "source" array of 'double' values.
     y:  ordinate "source" array of 'double' values.
     tension:  tension value for interpolation, use 0.0 as default vlaue:
               If tension=0 then a spline with tension reduces to a
               conventional piecewise cubic spline.  In the limits
               tension->+infinity and tension->-infinity, a spline with
               tension reduces to a piecewise linear (`broken line')
               interpolation.  To oversimplify a bit, 1.0/tension is the
               maximum abscissa range over which the spline likes to curve,
               at least when tension>0.  So increasing the tension far above
               zero tends to make the spline contain short curved sections,
               separated by sections that are almost straight.  The curved
               sections will be centered on the user-specified data points.
               The behavior of the spline when tension<0 is altogether
               different: it will tend to oscillate, though as
               tension->-infinity the oscillations are damped out.
     k:  boundary condition, use 1.0 as default value:  Appears in the two
               boundary conditions y''[0]=ky''[1] and y''[n]=ky''[n-1].
     xvals_arr:  array of "new" abscissa values to use with interpolation
                 ('double' values).
     num_xvals:  number of entries in 'xvals_arr'.
     **p_retvals_arr:  reference to ordinate "destination" array of 'double'
                       values generated via interpolation.
     *p_num_retvals:  reference to number of values returned in
                      'p_retvals_arr' (will be less than 'num_xvals'
                      if any new abscissa values are out of range).
   Returns:  NULL if successful; an error message string if not.
*/
char *evr_spline(int num_points, double *t, double *y, double tension,
                                 double k, double *xvals_arr, int num_xvals,
                                double **p_retvals_arr, int *p_num_retvals);

