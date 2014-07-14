#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

/*
 *   1/18/2006 -- [ET]  Renamed function to prevent name clashes with
 *                      other libraries.
 */

#include <stdio.h>
#include <stdlib.h>          /* for 'exit()'; added 8/28/2001 -- [ET] */

void
evr_regerror(char *s)
{
#ifdef ERRAVAIL
	error("regexp: %s", s);
#else
	fprintf(stderr, "regexp(3): %s", s);
	exit(1);
#endif
	/* NOTREACHED */
}
