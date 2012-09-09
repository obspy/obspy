/*
 * Definitions etc. for regexp(3) routines.
 *
 * Caveat:  this is V8 regexp(3) [actually, a reimplementation thereof],
 * not the System V one.
 */
/*
 *   8/28/2001 -- [ET]  Added parameter lists to function declarations.
 *   1/18/2006 -- [ET]  Renamed functions to prevent name clashes with
 *                      other libraries.
 */
#define NSUBEXP  10
typedef struct regexp {
	char *startp[NSUBEXP];
	char *endp[NSUBEXP];
	char regstart;		/* Internal use only. */
	char reganch;		/* Internal use only. */
	char *regmust;		/* Internal use only. */
	int regmlen;		/* Internal use only. */
	char program[1];	/* Unwarranted chumminess with compiler. */
} regexp;

regexp *evr_regcomp(char *exp);
int evr_regexec(regexp *prog, char *string);
void evr_regsub(regexp *prog, char *source, char *dest);
void evr_regerror(char *s);
