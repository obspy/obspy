/***************************************************************************
 * selection.c:
 *
 * Generic routines to manage selection lists.
 *
 * Written by Chad Trabant unless otherwise noted
 *   IRIS Data Management Center
 *
 * modified: 2010.068
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>

#include "libmseed.h"

static int ms_globmatch (char *string, char *pattern);


/***************************************************************************
 * ms_matchselect:
 *
 * Test the specified parameters for a matching selection entry.  The
 * srcname parameter may contain globbing characters.  The NULL value
 * (matching any times) for the start and end times is HPTERROR.
 *
 * Return Selections pointer to matching entry on successful match and
 * NULL for no match or error.
 ***************************************************************************/
Selections *
ms_matchselect (Selections *selections, char *srcname, hptime_t starttime,
		hptime_t endtime, SelectTime **ppselecttime)
{
  Selections *findsl = NULL;
  SelectTime *findst = NULL;
  SelectTime *matchst = NULL;
  
  if ( selections )
    {
      findsl = selections;
      while ( findsl )
	{
	  if ( ms_globmatch (srcname, findsl->srcname) )
	    {
	      findst = findsl->timewindows;              
	      while ( findst )
		{
		  if ( starttime != HPTERROR && findst->starttime != HPTERROR &&
		       (starttime < findst->starttime && ! (starttime <= findst->starttime && endtime >= findst->starttime)) )
		    { findst = findst->next; continue; }
		  else if ( endtime != HPTERROR && findst->endtime != HPTERROR &&
			    (endtime > findst->endtime && ! (starttime <= findst->endtime && endtime >= findst->endtime)) )
		    { findst = findst->next; continue; }
		  
		  matchst = findst;
		  break;
		}
	    }
	  
	  if ( matchst )
	    break;
	  else
	    findsl = findsl->next;
	}
    }
  
  if ( ppselecttime )
    *ppselecttime = matchst;
  
  return ( matchst ) ? findsl : NULL;
} /* End of ms_matchselect() */


/***************************************************************************
 * msr_matchselect:
 *
 * A simple wrapper for calling ms_matchselect() using details from a
 * MSRecord struct.
 *
 * Return Selections pointer to matching entry on successful match and
 * NULL for no match or error.
 ***************************************************************************/
Selections *
msr_matchselect (Selections *selections, MSRecord *msr, SelectTime **ppselecttime)
{
  char srcname[50];
  hptime_t endtime;
  
  if ( ! selections || ! msr )
    return NULL;
  
  msr_srcname (msr, srcname, 1);
  endtime = msr_endtime (msr);
  
  return ms_matchselect (selections, srcname, msr->starttime, endtime,
			 ppselecttime);
} /* End of msr_matchselect() */


/***************************************************************************
 * ms_addselect:
 *
 * Add select parameters to a specified selection list.  The srcname
 * argument may contain globbing parameters.  The NULL value (matching
 * any value) for the start and end times is HPTERROR.
 *
 * Return 0 on success and -1 on error.
 ***************************************************************************/
int
ms_addselect (Selections **ppselections, char *srcname,
	      hptime_t starttime, hptime_t endtime)
{
  Selections *newsl = NULL;
  SelectTime *newst = NULL;
  
  if ( ! ppselections || ! srcname )
    return -1;
  
  /* Allocate new SelectTime and populate */
  if ( ! (newst = (SelectTime *) calloc (1, sizeof(SelectTime))) )
    {
      ms_log (2, "Cannot allocate memory\n");
      return -1;
    }
  
  newst->starttime = starttime;
  newst->endtime = endtime;
  
  /* Add new Selections struct to begining of list */
  if ( ! *ppselections )
    {
      /* Allocate new Selections and populate */
      if ( ! (newsl = (Selections *) calloc (1, sizeof(Selections))) )
	{
	  ms_log (2, "Cannot allocate memory\n");
	  return -1;
	}
      
      strncpy (newsl->srcname, srcname, sizeof(newsl->srcname));
      
      /* Add new Selections struct as first in list */
      *ppselections = newsl;
      newsl->timewindows = newst;
    }
  else
    {
      Selections *findsl = *ppselections;
      Selections *matchsl = 0;
      
      /* Search for matching Selectlink entry */
      while ( findsl )
	{
	  if ( ! strcmp (findsl->srcname, srcname) )
	    {
	      matchsl = findsl;
	      break;
	    }
	  
	  findsl = findsl->next;
	}
      
      if ( matchsl )
	{
	  /* Add time window selection to beginning of window list */
	  newst->next = matchsl->timewindows;
	  matchsl->timewindows = newst;
	}
      else
	{
	  /* Allocate new Selections and populate */
	  if ( ! (newsl = (Selections *) calloc (1, sizeof(Selections))) )
	    {
	      ms_log (2, "Cannot allocate memory\n");
	      return -1;
	    }
	  
	  strncpy (newsl->srcname, srcname, sizeof(newsl->srcname));
	  
	  /* Add new Selections to beginning of list */
	  newsl->next = *ppselections;
	  *ppselections = newsl;
	  newsl->timewindows = newst;
	}
    }
  
  return 0;
} /* End of ms_addselect() */


/***************************************************************************
 * ms_addselect_comp:
 *
 * Add select parameters to a specified selection list based on
 * separate name components.  The network, station, location, channel
 * and quality arguments may contain globbing parameters.  The NULL
 * value (matching any value) for the start and end times is HPTERROR.
 *
 * If any of the naming parameters are not supplied (pointer is NULL)
 * a wildcard for all matches is substituted.  As a special case, if
 * the location ID (loc) is set to "--" to match a space-space/blank
 * ID it will be translated to an empty string to match libmseed's
 * notation.
 *
 * Return 0 on success and -1 on error.
 ***************************************************************************/
int
ms_addselect_comp (Selections **ppselections, char *net, char* sta, char *loc,
		   char *chan, char *qual, hptime_t starttime, hptime_t endtime)
{
  char srcname[100];
  char selnet[20];
  char selsta[20];
  char selloc[20];
  char selchan[20];
  char selqual[20];
  
  if ( ! ppselections )
    return -1;
  
  if ( net )
    {
      strncpy (selnet, net, sizeof(selnet));
      selnet[sizeof(selnet)-1] = '\0';
    }
  else
    strcpy (selnet, "*");
  
  if ( sta )
    {
      strncpy (selsta, sta, sizeof(selsta));
      selsta[sizeof(selsta)-1] = '\0';
    }
  else
    strcpy (selsta, "*");
  
  if ( loc )
    {
      /* Test for special case blank location ID */
      if ( ! strcmp (loc, "--") )
	selloc[0] = '\0';
      else
	{
	  strncpy (selloc, loc, sizeof(selloc));
	  selloc[sizeof(selloc)-1] = '\0';
	}
    }
  else
    strcpy (selloc, "*");
  
  if ( chan )
    {
      strncpy (selchan, chan, sizeof(selchan));
      selchan[sizeof(selchan)-1] = '\0';
    }
  else
    strcpy (selchan, "*");
  
  if ( qual )
    {
      strncpy (selqual, qual, sizeof(selqual));
      selqual[sizeof(selqual)-1] = '\0';
    }
  else
    strcpy (selqual, "?");
  
  /* Create the srcname globbing match for this entry */
  snprintf (srcname, sizeof(srcname), "%s_%s_%s_%s_%s",
	    selnet, selsta, selloc, selchan, selqual);
  
  /* Add selection to list */
  if ( ms_addselect (ppselections, srcname, starttime, endtime) )
    return -1;
  
  return 0;
} /* End of ms_addselect_comp() */


/***************************************************************************
 * ms_readselectionsfile:
 *
 * Read a list of data selections from a file and them to the
 * specified selections list.  On errors this routine will leave
 * allocated memory unreachable (leaked), it is expected that this is
 * a program failing condition.
 *
 * As a special case if the filename is "-", selection lines will be
 * read from stdin.
 *
 * Returns count of selections added on success and -1 on error.
 ***************************************************************************/
int
ms_readselectionsfile (Selections **ppselections, char *filename)
{
  FILE *fp;
  hptime_t starttime;
  hptime_t endtime;
  char selectline[200];
  char *selnet;
  char *selsta;
  char *selloc;
  char *selchan;
  char *selqual;
  char *selstart;
  char *selend;
  char *cp;
  char next;
  int selectcount = 0;
  int linecount = 0;
  
  if ( ! ppselections || ! filename )
    return -1;
  
  if ( strcmp (filename, "-" ) )
    {
      if ( ! (fp = fopen(filename, "rb")) )
	{
	  ms_log (2, "Cannot open file %s: %s\n", filename, strerror(errno));
	  return -1;
	}
    }
  else
    {
      /* Use stdin as special case */
      fp = stdin;
    }
  
  while ( fgets (selectline, sizeof(selectline)-1, fp) )
    {
      selnet = 0;
      selsta = 0;
      selloc = 0;
      selchan = 0;
      selqual = 0;
      selstart = 0;
      selend = 0;
      
      linecount++;
      
      /* Guarantee termination */
      selectline[sizeof(selectline)-1] = '\0';
      
      /* End string at first newline character if any */
      if ( (cp = strchr(selectline, '\n')) )
        *cp = '\0';
      
      /* Skip empty lines */
      if ( ! strlen (selectline) )
        continue;
      
      /* Skip comment lines */
      if ( *selectline == '#' )
        continue;
      
      /* Parse: identify components of selection and terminate */
      cp = selectline;
      next = 1;
      while ( *cp )
	{
	  if ( *cp == ' ' || *cp == '\t' ) { *cp = '\0'; next = 1; }
	  else if ( *cp == '#' ) { *cp = '\0'; break; }
	  else if ( next && ! selnet ) { selnet = cp; next = 0; }
	  else if ( next && ! selsta ) { selsta = cp; next = 0; }
	  else if ( next && ! selloc ) { selloc = cp; next = 0; }
	  else if ( next && ! selchan ) { selchan = cp; next = 0; }
	  else if ( next && ! selqual ) { selqual = cp; next = 0; }
	  else if ( next && ! selstart ) { selstart = cp; next = 0; }
	  else if ( next && ! selend ) { selend = cp; next = 0; }
	  else if ( next ) { *cp = '\0'; break; }
	  cp++;
	}
      
      /* Skip line if network, station, location and channel are not defined */
      if ( ! selnet || ! selsta || ! selloc || ! selchan )
	{
	  ms_log (2, "[%s] Skipping data selection line number %d\n", filename, linecount);
	  continue;
	}
      
      if ( selstart )
	{
	  starttime = ms_seedtimestr2hptime (selstart);
	  if ( starttime == HPTERROR )
	    {
	      ms_log (2, "Cannot convert data selection start time (line %d): %s\n", linecount, selstart);
	      return -1;
	    }
	}
      else
	{
	  starttime = HPTERROR;
	}
      
      if ( selend )
	{
	  endtime = ms_seedtimestr2hptime (selend);
	  if ( endtime == HPTERROR )
	    {
	      ms_log (2, "Cannot convert data selection end time (line %d): %s\n",  linecount, selend);
	      return -1;
	    }
	}
      else
	{
	  endtime = HPTERROR;
	}
      
      /* Add selection to list */
      if ( ms_addselect_comp (ppselections, selnet, selsta, selloc, selchan, selqual, starttime, endtime) )
	{
	  ms_log (2, "[%s] Error adding selection on line %d\n", filename, linecount);
	  return -1;
	}
      
      selectcount++;
    }
  
  if ( fp != stdin )
    fclose (fp);
  
  return selectcount;
} /* End of ms_readselectionsfile() */


/***************************************************************************
 * ms_freeselections:
 *
 * Free all memory associated with a Selections struct.
 ***************************************************************************/
void
ms_freeselections ( Selections *selections )
{
  Selections *select;
  Selections *selectnext;
  SelectTime *selecttime;
  SelectTime *selecttimenext;
  
  if ( selections )
    {
      select = selections;
      
      while ( select )
	{
	  selectnext = select->next;
	  
	  selecttime = select->timewindows;
	  
	  while ( selecttime )
	    {
	      selecttimenext = selecttime->next;
	      
	      free (selecttime);
	      
	      selecttime = selecttimenext;
	    }
	  
	  free (select);
	  
	  select = selectnext;
	}
    }

} /* End of ms_freeselections() */


/***************************************************************************
 * ms_printselections:
 *
 * Print the selections list using the ms_log() facility.
 ***************************************************************************/
void
ms_printselections ( Selections *selections )
{
  Selections *select;
  SelectTime *selecttime;
  char starttime[50];
  char endtime[50];
  
  if ( ! selections )
    return;
  
  select = selections;
  while ( select )
    {
      ms_log (0, "Selection: %s\n", select->srcname);
      
      selecttime = select->timewindows;
      while ( selecttime )
	{
	  if ( selecttime->starttime != HPTERROR )
	    ms_hptime2seedtimestr (selecttime->starttime, starttime, 1);
	  else
	    strncpy (starttime, "No start time", sizeof(starttime)-1);
	  
	  if ( selecttime->endtime != HPTERROR )
	    ms_hptime2seedtimestr (selecttime->endtime, endtime, 1);
	  else
	    strncpy (endtime, "No end time", sizeof(endtime)-1);
	  
	  ms_log (0, "  %30s  %30s\n", starttime, endtime);
	  
	  selecttime = selecttime->next;
	}
      
      select = select->next;
    }
} /* End of ms_printselections() */


/***********************************************************************
 * robust glob pattern matcher
 * ozan s. yigit/dec 1994
 * public domain
 *
 * glob patterns:
 *	*	matches zero or more characters
 *	?	matches any single character
 *	[set]	matches any character in the set
 *	[^set]	matches any character NOT in the set
 *		where a set is a group of characters or ranges. a range
 *		is written as two characters seperated with a hyphen: a-z denotes
 *		all characters between a to z inclusive.
 *	[-set]	set matches a literal hypen and any character in the set
 *	[]set]	matches a literal close bracket and any character in the set
 *
 *	char	matches itself except where char is '*' or '?' or '['
 *	\char	matches char, including any pattern character
 *
 * examples:
 *	a*c		ac abc abbc ...
 *	a?c		acc abc aXc ...
 *	a[a-z]c		aac abc acc ...
 *	a[-a-z]c	a-c aac abc ...
 *
 * Revision 1.4  2004/12/26  12:38:00  ct
 * Changed function name (amatch -> globmatch), variables and
 * formatting for clarity.  Also add matching header globmatch.h.
 *
 * Revision 1.3  1995/09/14  23:24:23  oz
 * removed boring test/main code.
 *
 * Revision 1.2  94/12/11  10:38:15  oz
 * charset code fixed. it is now robust and interprets all
 * variations of charset [i think] correctly, including [z-a] etc.
 * 
 * Revision 1.1  94/12/08  12:45:23  oz
 * Initial revision
 ***********************************************************************/

#define GLOBMATCH_TRUE    1
#define GLOBMATCH_FALSE   0
#define GLOBMATCH_NEGATE '^'       /* std char set negation char */

/***********************************************************************
 * ms_globmatch:
 *
 * Check if a string matches a globbing pattern.
 *
 * Return 0 if string does not match pattern and non-zero otherwise.
 **********************************************************************/
static int
ms_globmatch (char *string, char *pattern)
{
  int negate;
  int match;
  int c;
  
  while ( *pattern )
    {
      if ( !*string && *pattern != '*' )
	return GLOBMATCH_FALSE;
      
      switch ( c = *pattern++ )
	{
	  
	case '*':
	  while ( *pattern == '*' )
	    pattern++;
	  
	  if ( !*pattern )
	    return GLOBMATCH_TRUE;
	  
	  if ( *pattern != '?' && *pattern != '[' && *pattern != '\\' )
	    while ( *string && *pattern != *string )
	      string++;
	  
	  while ( *string )
	    {
	      if ( ms_globmatch(string, pattern) )
		return GLOBMATCH_TRUE;
	      string++;
	    }
	  return GLOBMATCH_FALSE;
	  
	case '?':
	  if ( *string )
	    break;
	  return GLOBMATCH_FALSE;
	  
	  /* set specification is inclusive, that is [a-z] is a, z and
	   * everything in between. this means [z-a] may be interpreted
	   * as a set that contains z, a and nothing in between.
	   */
	case '[':
	  if ( *pattern != GLOBMATCH_NEGATE )
	    negate = GLOBMATCH_FALSE;
	  else
	    {
	      negate = GLOBMATCH_TRUE;
	      pattern++;
	    }
	  
	  match = GLOBMATCH_FALSE;
	  
	  while ( !match && (c = *pattern++) )
	    {
	      if ( !*pattern )
		return GLOBMATCH_FALSE;
	      
	      if ( *pattern == '-' ) 	/* c-c */
		{
		  if ( !*++pattern )
		    return GLOBMATCH_FALSE;
		  if ( *pattern != ']' )
		    {
		      if ( *string == c || *string == *pattern ||
			   ( *string > c && *string < *pattern ) )
			match = GLOBMATCH_TRUE;
		    }
		  else
		    {		/* c-] */
		      if ( *string >= c )
			match = GLOBMATCH_TRUE;
		      break;
		    }
		}
	      else			/* cc or c] */
		{
		  if ( c == *string )
		    match = GLOBMATCH_TRUE;
		  if ( *pattern != ']' )
		    {
		      if ( *pattern == *string )
			match = GLOBMATCH_TRUE;
		    }
		  else
		    break;
		}
	    } 
	  
	  if ( negate == match )
	    return GLOBMATCH_FALSE;
	  
	  /*
	   * if there is a match, skip past the charset and continue on
	   */
	  while ( *pattern && *pattern != ']' )
	    pattern++;
	  if ( !*pattern++ )	/* oops! */
	    return GLOBMATCH_FALSE;
	  break;
	  
	case '\\':
	  if ( *pattern )
	    c = *pattern++;
	default:
	  if ( c != *string )
	    return GLOBMATCH_FALSE;
	  break;
	}
      
      string++;
    }
  
  return !*string;
}  /* End of ms_globmatch() */
