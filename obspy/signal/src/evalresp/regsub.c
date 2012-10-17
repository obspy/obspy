/*
 * regsub
 *
 *	Copyright (c) 1986 by University of Toronto.
 *	Written by Henry Spencer.  Not derived from licensed software.
 *
 *	Permission is granted to anyone to use this software for any
 *	purpose on any computer system, and to redistribute it freely,
 *	subject to the following restrictions:
 *
 *	1. The author is not responsible for the consequences of use of
 *		this software, no matter how awful, even if they arise
 *		from defects in it.
 *
 *	2. The origin of this software must not be misrepresented, either
 *		by explicit claim or by omission.
 *
 *	3. Altered versions must be plainly marked as such, and must not
 *		be misrepresented as being the original software.
 *
 *   1/18/2006 -- [ET]  Renamed function to prevent name clashes with
 *                      other libraries.
 */
#include <stdio.h>
#include <string.h>          /* added 8/28/2001 -- [ET] */
#include "regexp.h"
#include "regmagic.h"

#ifndef CHARBITS
#define	UCHARAT(p)	((int)*(unsigned char *)(p))
#else
#define	UCHARAT(p)	((int)*(p)&CHARBITS)
#endif

/*
 - evr_regsub - perform substitutions after a regexp match
 */
void
evr_regsub(prog, source, dest)
regexp *prog;
char *source;
char *dest;
{
	register char *src;
	register char *dst;
	register char c;
	register int no;
	register int len;

	if (prog == NULL || source == NULL || dest == NULL) {
		evr_regerror("NULL parm to evr_regsub");
		return;
	}
	if (UCHARAT(prog->program) != MAGIC) {
		evr_regerror("damaged regexp fed to evr_regsub");
		return;
	}

	src = source;
	dst = dest;
	while ((c = *src++) != '\0') {
		if (c == '&')
			no = 0;
		else if (c == '\\' && '0' <= *src && *src <= '9')
			no = *src++ - '0';
		else
			no = -1;

		if (no < 0)	/* Ordinary character. */
			*dst++ = c;
		else if (prog->startp[no] != NULL && prog->endp[no] != NULL) {
			len = prog->endp[no] - prog->startp[no];
			(void) strncpy(dst, prog->startp[no], len);
			dst += len;
			if (*(dst-1) == '\0') {		/* strncpy hit NUL. */
				evr_regerror("damaged match string");
				return;
			}
		}
	}
	*dst++ = '\0';
}
