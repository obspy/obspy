/* evr_spline.c:  Cubic spline function for 'evalresp'.

  10/14/2005 -- [ET]  Initial version, based on 'spline' module
                      from GNU 'plotutils' package
                      (http://www.gnu.org/software/plotutils)


   Original notes from GNU 'spline' module:

   Written by Robert S. Maier
   <rsm@math.arizona.edu>, based on earlier work by Rich Murphey.
   Copyright (C) 1989-1999 Free Software Foundation, Inc.

   References:

   D. Kincaid and [E.] W. Cheney, Numerical Analysis, Brooks/Cole,
   2nd. ed., 1996, Section 6.4.

   C. de Boor, A Practical Guide to Splines, Springer-Verlag, 1978,
   Chapter 4.

   A. K. Cline, "Scalar and Planar-Valued Curve Fitting Using Splines under
   Tension", Communications of the ACM 17 (1974), 218-223.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "evresp.h"

/* Minimum value for magnitude of x, for such functions as x-sinh(x),
   x-tanh(x), x-sin(x), and x-tan(x) to have acceptable accuracy.  If the
   magnitude of x is smaller than this value, these functions of x will be
   computed via power series to accuracy O(x**6). */
#define TRIG_ARG_MIN 0.001

/* Maximum value for magnitude of x, beyond which we approximate
   x/sinh(x) and x/tanh(x) by |x|exp(-|x|). */
#define TRIG_ARG_MAX 50.0

typedef enum { false = 0, true = 1 } bool;

/* forward references */
double interpolate(int n, double *t, double *y, double *z, double x,
                                             double tension, bool periodic);
char *fit(int n, double *t, double *y, double *z, double k, double tension,
                                                             bool periodic);
bool is_monotonic(int n, double *t);
double quotient_sin_func(double x, double y);
double quotient_sinh_func(double x, double y);
double sin_func(double x);
double sinh_func(double x);
double tan_func(double x);
double tanh_func(double x);
extern void *spl_malloc(size_t size);

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
                                 double **p_retvals_arr, int *p_num_retvals)
{
  int used = num_points - 1;
  int range_count = 0;		/* number of req'd datapoints out of range */
  int lastval = 0;		/* last req'd point = 1st/last data point? */
  int i;
  double x, *z;
  double last_t;
  char *retstr;

  if (num_points < 2 || num_xvals <= 0)
    {    /* less than 2 "source" datapoints or no "new" abscissa points */
      *p_num_retvals = 0;        /* don't output anything (null dataset) */
      return NULL;
    }

  if (num_points <= 2)
    {
      k = 0.0;
    }

  if (!is_monotonic (used, t))
    return "Abscissa values not monotonic";

                                /* create array for 2nd derivatives */
  z = (double *)spl_malloc (sizeof(double) * num_points);

  /* compute z, array of 2nd derivatives at each knot */
  if((retstr=fit(used, t, y, z, k, tension, false)) != NULL)
    {
      free(z);                 /* free array for 2nd derivatives */
      return retstr;           /* if error string returned then return it */
    }

  last_t = xvals_arr[num_xvals-1];     /* last value in 'x' array */
  if (last_t == t[0])
    lastval = 1;
  else if (last_t == t[used])
    lastval = 2;

                             /* allocate array to be returned */
  *p_retvals_arr = (double *)spl_malloc (sizeof(double) * num_xvals);
  *p_num_retvals = 0;        /* initialize count */

  for (i = 0; i < num_xvals; ++i)
    {    /* for each value in "new" abscissa array */
      x = xvals_arr[i];          /* get value from array */

      if (i == num_xvals-1)
	{
	  /* avoid numerical fuzz */
	  if (lastval == 1)	 /* left end of input */
	    x = t[0];
	  else if (lastval == 2) /* right end of input */
	    x = t[used];
	}

      if ((x - t[0]) * (x - t[used]) <= 0)
	{               /* calculate value, enter into array, inc index */
          (*p_retvals_arr)[(*p_num_retvals)++] = interpolate(
                                          used, t, y, z, x, tension, false);
	}
      else
	range_count++;
    }
  free(z);                       /* free array for 2nd derivatives */

  if(range_count > 0)
    {
      return "One or more requested points could not be computed (out of data range)";
    }

  return NULL;
}


/* fit() computes the array z[] of second derivatives at the knots, i.e.,
   internal data points.  The abscissa array t[] and the ordinate array y[]
   are specified.  On entry, have n+1 >= 2 points in the t, y, z arrays,
   numbered 0..n.  The knots are numbered 1..n-1 as in Kincaid and Cheney.
   In the periodic case, the final knot, i.e., (t[n-1],y[n-1]), has the
   property that y[n-1]=y[0]; moreover, y[n]=y[1].  The number of points
   supplied by the user was n+1 in the non-periodic case, and n in the
   periodic case.  When this function is called, n>=1 in the non-periodic
   case, and n>=2 in the periodic case.
   Returns:  NULL if successful; an error message string if not. */

/* Algorithm: the n-1 by n-1 tridiagonal matrix equation for the vector of
   2nd derivatives at the knots is reduced to upper diagonal form.  At that
   point the diagonal entries (pivots) of the upper diagonal matrix are in
   the vector u[], and the vector on the right-hand side is v[].  That is,
   the equation is of the form Ay'' = v, where a_(ii) = u[i], and a_(i,i+1)
   = alpha[i].  Here i=1..n-1 indexes the set of knots.  The matrix
   equation is solved by back-substitution for y''[], i.e., for z[]. */

char *fit(int n, double *t, double *y, double *z, double k, double tension,
           bool periodic)
{
  double *h, *b, *u, *v, *alpha, *beta;
  double *uu = NULL, *vv = NULL, *s = NULL;
  int i;

  if (n == 1)			/* exactly 2 points, use straight line */
    {
      z[0] = z[1] = 0.0;
      return NULL;
    }

  h = (double *)spl_malloc (sizeof(double) * n);
  b = (double *)spl_malloc (sizeof(double) * n);
  u = (double *)spl_malloc (sizeof(double) * n);
  v = (double *)spl_malloc (sizeof(double) * n);
  alpha = (double *)spl_malloc (sizeof(double) * n);
  beta = (double *)spl_malloc (sizeof(double) * n);

  if (periodic)
    {
      s = (double *)spl_malloc (sizeof(double) * n); 
      uu = (double *)spl_malloc (sizeof(double) * n); 
      vv = (double *)spl_malloc (sizeof(double) * n); 
    }

  for (i = 0; i <= n - 1 ; ++i)
    {
      h[i] = t[i + 1] - t[i];
      b[i] = 6.0 * (y[i + 1] - y[i]) / h[i]; /* for computing RHS */
    }

  if (tension < 0.0)		/* must rule out sin(tension * h[i]) = 0 */
    {
      for (i = 0; i <= n - 1 ; ++i)
	if (sin (tension * h[i]) == 0.0)
	  {
            return "Specified negative tension value is singular";
	  }
    }
  if (tension == 0.0)
    {
      for (i = 0; i <= n - 1 ; ++i)
	{
	  alpha[i] = h[i];	/* off-diagonal = alpha[i] to right */
	  beta[i] = 2.0 * h[i];	/* diagonal = beta[i-1] + beta[i] */
	}
    }
  else
    if (tension > 0.0)
      /* `positive' (really real) tension, use hyperbolic trig funcs */
      {
	for (i = 0; i <= n - 1 ; ++i)
	  {
	    double x = tension * h[i];
	    double xabs = (x < 0.0 ? -x : x);

	    if (xabs < TRIG_ARG_MIN)
	      /* hand-compute (6/x^2)(1-x/sinh(x)) and (3/x^2)(x/tanh(x)-1)
                 to improve accuracy; here `x' is tension * h[i] */
	      {
		alpha[i] = h[i] * sinh_func(x);
		beta[i] = 2.0 * h[i] * tanh_func(x);
	      }
	    else if (xabs > TRIG_ARG_MAX)
	      /* in (6/x^2)(1-x/sinh(x)) and (3/x^2)(x/tanh(x)-1),
		 approximate x/sinh(x) and x/tanh(x) by 2|x|exp(-|x|)
		 and |x|, respectively */
	      {
		int sign = (x < 0.0 ? -1 : 1);

		alpha[i] = ((6.0 / (tension * tension))
			   * ((1.0 / h[i]) - tension * 2 * sign * exp(-xabs)));
		beta[i] = ((6.0 / (tension * tension))
			   * (tension - (1.0 / h[i])));
	      }
	    else
	      {
		alpha[i] = ((6.0 / (tension * tension))
			    * ((1.0 / h[i]) - tension / sinh(x)));
		beta[i] = ((6.0 / (tension * tension))
			   * (tension / tanh(x) - (1.0 / h[i])));
	      }
	  }
      }
    else				/* tension < 0 */
      /* `negative' (really imaginary) tension,  use circular trig funcs */
      {
	for (i = 0; i <= n - 1 ; ++i)
	  {
	    double x = tension * h[i];
	    double xabs = (x < 0.0 ? -x : x);

	    if (xabs < TRIG_ARG_MIN)
	      /* hand-compute (6/x^2)(1-x/sin(x)) and (3/x^2)(x/tan(x)-1)
                 to improve accuracy; here `x' is tension * h[i] */
	      {
		alpha[i] = h[i] * sin_func(x);
		beta[i] = 2.0 * h[i] * tan_func(x);
	      }
	    else
	      {
		alpha[i] = ((6.0 / (tension * tension))
		           * ((1.0 / h[i]) - tension / sin(x)));
		beta[i] = ((6.0 / (tension * tension))
			   * (tension / tan(x) - (1.0 / h[i])));
	      }
	  }
      }
  
  if (!periodic && n == 2)
      u[1] = beta[0] + beta[1] + 2 * k * alpha[0];
  else
    u[1] = beta[0] + beta[1] + k * alpha[0];

  v[1] = b[1] - b[0];
  
  if (u[1] == 0.0)
    {
      return "As posed, problem of computing spline is singular";
    }

  if (periodic)
    {
      s[1] = alpha[0];
      uu[1] = 0.0;
      vv[1] = 0.0;
    }

  for (i = 2; i <= n - 1 ; ++i)
    {
      u[i] = (beta[i] + beta[i - 1]
	      - alpha[i - 1] * alpha[i - 1] / u[i - 1]
	      + (i == n - 1 ? k * alpha[n - 1] : 0.0));

      if (u[i] == 0.0)
	{
          return "As posed, problem of computing spline is singular";
	}


      v[i] = b[i] - b[i - 1] - alpha[i - 1] * v[i - 1] / u[i - 1];

      if (periodic)
	{
	  s[i] = - s[i-1] * alpha[i-1] / u[i-1];
	  uu[i] = uu[i-1] - s[i-1] * s[i-1] / u[i-1];
	  vv[i] = vv[i-1] - v[i-1] * s[i-1] / u[i-1];
	}
    }
      
  if (!periodic)
    {
      /* fill in 2nd derivative array */
      z[n] = 0.0;
      for (i = n - 1; i >= 1; --i)
	z[i] = (v[i] - alpha[i] * z[i + 1]) / u[i];
      z[0] = 0.0;
      
      /* modify to include boundary condition */
      z[0] = k * z[1];
      z[n] = k * z[n - 1];
    }
  else		/* periodic */
    {
      z[n-1] = (v[n-1] + vv[n-1]) / (u[n-1] + uu[n-1] + 2 * s[n-1]);
      for (i = n - 2; i >= 1; --i)
	z[i] = ((v[i] - alpha[i] * z[i + 1]) - s[i] * z[n-1]) / u[i];

      z[0] = z[n-1];
      z[n] = z[1];
    }

  if (periodic)
    {
      free (vv);
      free (uu);
      free (s);
    }
  free (beta);
  free (alpha);
  free (v);
  free (u);
  free (b);
  free (h);

  return NULL;
}

/* interpolate() computes an approximate ordinate value for a given
   abscissa value, given an array of data points (stored in t[] and y[],
   containing abscissa and ordinate values respectively), and z[], the
   array of 2nd derivatives at the knots (i.e. internal data points).

   On entry, have n+1 >= 2 points in the t, y, z arrays, numbered 0..n.
   The number of knots (i.e. internal data points) is n-1; they are
   numbered 1..n-1 as in Kincaid and Cheney.  In the periodic case, the
   final knot, i.e., (t[n-1],y[n-1]), has the property that y[n-1]=y[0];
   also, y[n]=y[1].  The number of data points supplied by the user was n+1
   in the non-periodic case, and n in the periodic case.  When this
   function is called, n>=1 in the non-periodic case, and n>=2 in the
   periodic case. */

double interpolate(int n, double *t, double *y, double *z, double x,
	     double tension, bool periodic)
{
  double diff, updiff, reldiff, relupdiff, h;
  double value;
  int is_ascending = (t[n-1] < t[n]);
  int i = 0, k;

  /* in periodic case, map x to t[0] <= x < t[n] */
  if (periodic && (x - t[0]) * (x - t[n]) > 0.0)
    x -= ((int)(floor( (x - t[0]) / (t[n] - t[0]) )) * (t[n] - t[0]));

  /* do binary search to find interval */
  for (k = n - i; k > 1;)
    {
      if (is_ascending ? x >= t[i + (k>>1)] : x <= t[i + (k>>1)])
	{
	  i = i + (k>>1);
	  k = k - (k>>1);
	}
      else
	k = k>>1;
    }

  /* at this point, x is between t[i] and t[i+1] */
  h = t[i + 1] - t[i];
  diff = x - t[i];
  updiff = t[i+1] - x;
  reldiff = diff / h;
  relupdiff = updiff / h;

  if (tension == 0.0)
  /* evaluate cubic polynomial in nested form */
    value =  y[i]
      + diff
	* ((y[i + 1] - y[i]) / h - h * (z[i + 1] + z[i] * 2.0) / 6.0
	   + diff * (0.5 * z[i] + diff * (z[i + 1] - z[i]) / (6.0 * h)));

  else if (tension > 0.0)
    /* `positive' (really real) tension, use sinh's */
    {
      if (fabs(tension * h) < TRIG_ARG_MIN)
	/* hand-compute (6/y^2)(sinh(xy)/sinh(y) - x) to improve accuracy;
	   here `x' means reldiff or relupdiff and `y' means tension*h */
	value = (y[i] * relupdiff + y[i+1] * reldiff
		 + ((z[i] * h * h / 6.0) 
		    * quotient_sinh_func (relupdiff, tension * h))
		 + ((z[i+1] * h * h / 6.0) 
		    * quotient_sinh_func (reldiff, tension * h)));
      else if (fabs(tension * h) > TRIG_ARG_MAX)
	/* approximate 1/sinh(y) by 2 sgn(y) exp(-|y|) */
	{
	  int sign = (h < 0.0 ? -1 : 1);

	  value = (((z[i] * (exp (tension * updiff - sign * tension * h) 
			     + exp (-tension * updiff - sign * tension * h))
		     + z[i + 1] * (exp (tension * diff - sign * tension * h) 
				   + exp (-tension * diff - sign * tension*h)))
		    * (sign / (tension * tension)))
		   + (y[i] - z[i] / (tension * tension)) * (updiff / h)
		   + (y[i + 1] - z[i + 1] / (tension * tension)) * (diff / h));
	}
      else
	value = (((z[i] * sinh (tension * updiff) 
		   + z[i + 1] * sinh (tension * diff))
		  / (tension * tension * sinh (tension * h)))
		 + (y[i] - z[i] / (tension * tension)) * (updiff / h)
		 + (y[i + 1] - z[i + 1] / (tension * tension)) * (diff / h));
    }
  else
    /* `negative' (really imaginary) tension, use sin's */
    {
      if (fabs(tension * h) < TRIG_ARG_MIN)
	/* hand-compute (6/y^2)(sin(xy)/sin(y) - x) to improve accuracy;
	   here `x' means reldiff or relupdiff and `y' means tension*h */
	value = (y[i] * relupdiff + y[i+1] * reldiff
		 + ((z[i] * h * h / 6.0) 
		    * quotient_sin_func (relupdiff, tension * h))
		 + ((z[i+1] * h * h / 6.0) 
		    * quotient_sin_func (reldiff, tension * h)));
      else
	value = (((z[i] * sin (tension * updiff) 
		   + z[i + 1] * sin (tension * diff))
		  / (tension * tension * sin (tension * h)))
		 + (y[i] - z[i] / (tension * tension)) * (updiff / h)
		 + (y[i + 1] - z[i + 1] / (tension * tension)) * (diff / h));
    }
  
  return value;
}

/* is_monotonic() check whether an array of data points, read in by
   read_data(), has monotonic abscissa values. */
bool is_monotonic(int n, double *t)
{
  bool is_ascending;

  if (t[n-1] < t[n])
    is_ascending = true;
  else if (t[n-1] > t[n])
    is_ascending = false;
  else				/* equality */
    return false;

  while (n>0)
    {
      n--;
      if (is_ascending == true ? t[n] >= t[n+1] : t[n] <= t[n+1])
	return false;
    };
  return true;
}

/* Following four functions compute (6/x^2)(1-x/sinh(x)),
   (3/x^2)(x/tanh(x)-1), (6/x^2)(1-x/sin(x)), and (3/x^2)(x/tan(x)-1) via
   the first three terms of the appropriate power series.  They are used
   when |x|<TRIG_ARG_MIN, to avoid loss of significance.  Errors are
   O(x**6). */
double sinh_func(double x)
{
  /* use 1-(7/60)x**2+(31/2520)x**4 */
  return 1.0 - (7.0/60.0)*x*x + (31.0/2520.0)*x*x*x*x;
}

double tanh_func(double x)
{
  /* use 1-(1/15)x**2+(2/315)x**4 */
  return 1.0 - (1.0/15.0)*x*x + (2.0/315.0)*x*x*x*x;
}

double sin_func(double x)
{
  /* use -1-(7/60)x**2-(31/2520)x**4 */
  return -1.0 - (7.0/60.0)*x*x - (31.0/2520.0)*x*x*x*x;
}

double tan_func(double x)
{
  /* use -1-(1/15)x**2-(2/315)x**4 */
  return -1.0 - (1.0/15.0)*x*x - (2.0/315.0)*x*x*x*x;
}

/* Following two functions compute (6/y^2)(sinh(xy)/sinh(y)-x) and
   (6/y^2)(sin(xy)/sin(y)-x), via the first three terms of the appropriate
   power series in y.  They are used when |y|<TRIG_ARG_MIN, to avoid loss
   of significance.  Errors are O(y**6). */
double quotient_sinh_func(double x, double y)
{
  return ((x*x*x-x) + (x*x*x*x*x/20.0 - x*x*x/6.0 + 7.0*x/60.0)*(y*y)
	  + (x*x*x*x*x*x*x/840.0 - x*x*x*x*x/120.0 + 7.0*x*x*x/360.0
	     -31.0*x/2520.0)*(y*y*y*y));
}

double quotient_sin_func(double x, double y)
{
  return (- (x*x*x-x) + (x*x*x*x*x/20.0 - x*x*x/6.0 + 7.0*x/60.0)*(y*y)
	  - (x*x*x*x*x*x*x/840.0 - x*x*x*x*x/120.0 + 7.0*x*x*x/360.0
	     -31.0*x/2520.0)*(y*y*y*y));
}


void *spl_malloc(size_t length)
{
  void *p;
  p = (void *) malloc (length);

  if (p == (void *) NULL)
    error_exit(OUT_OF_MEMORY,"spl_malloc() failed");
  return p;
}

