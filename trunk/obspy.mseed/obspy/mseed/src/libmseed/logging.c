/***************************************************************************
 * logging.c
 *
 * Log handling routines for libmseed
 *
 * Chad Trabant
 * IRIS Data Management Center
 *
 * modified: 2010.253
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "libmseed.h"

void ms_loginit_main (MSLogParam *logp,
		      void (*log_print)(char*), const char *logprefix,
		      void (*diag_print)(char*), const char *errprefix);

int ms_log_main (MSLogParam *logp, int level, va_list *varlist);

/* Initialize the global logging parameters */
MSLogParam gMSLogParam = {NULL, NULL, NULL, NULL};


/***************************************************************************
 * ms_loginit:
 *
 * Initialize the global logging parameters.
 *
 * See ms_loginit_main() description for usage.
 ***************************************************************************/
void
ms_loginit (void (*log_print)(char*), const char *logprefix,
	    void (*diag_print)(char*), const char *errprefix)
{
  ms_loginit_main(&gMSLogParam, log_print, logprefix, diag_print, errprefix);
}  /* End of ms_loginit() */


/***************************************************************************
 * ms_loginit_l:
 *
 * Initialize MSLogParam specific logging parameters.  If the logging parameters
 * have not been initialized (log == NULL) new parameter space will
 * be allocated.
 *
 * See ms_loginit_main() description for usage.
 *
 * Returns a pointer to the created/re-initialized MSLogParam struct
 * on success and NULL on error.
 ***************************************************************************/
MSLogParam *
ms_loginit_l (MSLogParam *logp,
	      void (*log_print)(char*), const char *logprefix,
	      void (*diag_print)(char*), const char *errprefix)
{
  MSLogParam *llog;
  
  if ( logp == NULL )
    {
      llog = (MSLogParam *) malloc (sizeof(MSLogParam));
      
      if ( llog == NULL )
        {
          ms_log (2, "ms_loginit_l(): Cannot allocate memory\n");
          return NULL;
        }
      
      llog->log_print = NULL;
      llog->logprefix = NULL;
      llog->diag_print = NULL;
      llog->errprefix = NULL;
    }
  else
    {
      llog = logp;
    }
  
  ms_loginit_main (llog, log_print, logprefix, diag_print, errprefix);

  return llog;
}  /* End of ms_loginit_l() */


/***************************************************************************
 * ms_loginit_main:
 *
 * Initialize the logging subsystem.  Given values determine how ms_log()
 * and ms_log_l() emit messages.
 *
 * This function modifies the logging parameters in the passed MSLogParam.
 *
 * Any log/error printing functions indicated must except a single
 * argument, namely a string (const char *).  The ms_log() and
 * ms_log_r() functions format each message and then pass the result
 * on to the log/error printing functions.
 *
 * If the log/error prefixes have been set they will be pre-pended to the
 * message.
 *
 * Use NULL for the function pointers or the prefixes if they should not
 * be changed from previously set or default values.  The default behavior
 * of the logging subsystem is given in the example below.
 *
 * Example: ms_loginit_main (0, (void*)&printf, NULL, (void*)&printf, "error: ");
 ***************************************************************************/
void
ms_loginit_main (MSLogParam *logp,
		 void (*log_print)(char*), const char *logprefix,
		 void (*diag_print)(char*), const char *errprefix)
{
  if ( ! logp )
    return;

  if ( log_print )
    logp->log_print = log_print;
  
  if ( logprefix )
    {
      if ( strlen(logprefix) >= MAX_LOG_MSG_LENGTH )
	{
	  ms_log_l (logp, 2, 0, "log message prefix is too large\n");
	}
      else
	{
	  logp->logprefix = logprefix;
	}
    }
  
  if ( diag_print )
    logp->diag_print = diag_print;
  
  if ( errprefix )
    {
      if ( strlen(errprefix) >= MAX_LOG_MSG_LENGTH )
	{
	  ms_log_l (logp, 2, 0, "error message prefix is too large\n");
	}
      else
	{
	  logp->errprefix = errprefix;
	}
    }
  
  return;
}  /* End of ms_loginit_main() */


/***************************************************************************
 * ms_log:
 *
 * A wrapper to ms_log_main() that uses the global logging parameters.
 *
 * See ms_log_main() description for return values.
 ***************************************************************************/
int
ms_log (int level, ...)
{
  int retval;
  va_list varlist;
  
  va_start (varlist, level);

  retval = ms_log_main (&gMSLogParam, level, &varlist);

  va_end (varlist);

  return retval;
}  /* End of ms_log() */


/***************************************************************************
 * ms_log_l:
 *
 * A wrapper to ms_log_main() that uses the logging parameters in a
 * supplied MSLogParam.  If the supplied pointer is NULL the global logging
 * parameters will be used.
 *
 * See ms_log_main() description for return values.
 ***************************************************************************/
int
ms_log_l (MSLogParam *logp, int level, ...)
{
  int retval;
  va_list varlist;
  MSLogParam *llog;

  if ( ! logp )
    llog = &gMSLogParam;
  else
    llog = logp;
  
  va_start (varlist, level);
  
  retval = ms_log_main (llog, level, &varlist);

  va_end (varlist);

  return retval;
}  /* End of ms_log_l() */


/***************************************************************************
 * ms_log_main:
 *
 * A standard logging/printing routine.
 *
 * The function uses logging parameters specified in the supplied
 * MSLogParam.
 * 
 * This function expects 2+ arguments: message level, fprintf format,
 * and fprintf arguments. 
 *
 * Three levels are recognized:
 * 0  : Normal log messages, printed using log_print with logprefix
 * 1  : Diagnostic messages, printed using diag_print with logprefix
 * 2+ : Error messagess, printed using diag_print with errprefix
 *
 * This function builds the log/error message and passes to it as a
 * string (const char *) to the functions defined with ms_loginit() or
 * ms_loginit_l().  If the log/error printing functions have not been
 * defined messages will be printed with fprintf, log messages to
 * stdout and error messages to stderr.
 *
 * If the log/error prefix's have been set with ms_loginit() or
 * ms_loginit_l() they will be pre-pended to the message.
 *
 * All messages will be truncated to the MAX_LOG_MSG_LENGTH, this includes
 * any set prefix.
 *
 * Returns the number of characters formatted on success, and a
 * a negative value on error.
 ***************************************************************************/
int
ms_log_main (MSLogParam *logp, int level, va_list *varlist)
{
  static char message[MAX_LOG_MSG_LENGTH];
  int retvalue = 0;
  int presize;
  const char *format;
  
  if ( ! logp )
    {
      fprintf(stderr, "ms_log_main() called without specifying log parameters");
      return -1;
    }
  
  message[0] = '\0';

  format = va_arg (*varlist, const char *);

  if ( level >= 2 )  /* Error message */
    {
      if ( logp->errprefix != NULL )
        {
          strncpy (message, logp->errprefix, MAX_LOG_MSG_LENGTH);
        }
      else
        {
          strncpy (message, "Error: ", MAX_LOG_MSG_LENGTH);
        }
      
      presize = strlen(message);
      retvalue = vsnprintf (&message[presize],
   			    MAX_LOG_MSG_LENGTH - presize,
			    format, *varlist);
      
      message[MAX_LOG_MSG_LENGTH - 1] = '\0';

      if ( logp->diag_print != NULL )
        {
          logp->diag_print ((const char *) message);
        }
      else
        {
          fprintf(stderr, "%s", message);
        }
    }
  else if ( level == 1 )  /* Diagnostic message */
    {
      if ( logp->logprefix != NULL )
        {
          strncpy (message, logp->logprefix, MAX_LOG_MSG_LENGTH);
        }
      
      presize = strlen(message);
      retvalue = vsnprintf (&message[presize],
		            MAX_LOG_MSG_LENGTH - presize,
			    format, *varlist);
      
      message[MAX_LOG_MSG_LENGTH - 1] = '\0';
      
      if ( logp->diag_print != NULL )
        {
          logp->diag_print ((const char *) message);
        }
      else
        {
          fprintf(stderr, "%s", message);
        }
    }
  else if ( level == 0 )  /* Normal log message */
    {
      if ( logp->logprefix != NULL )
        {
          strncpy (message, logp->logprefix, MAX_LOG_MSG_LENGTH);
        }
      
      presize = strlen(message);
      retvalue = vsnprintf (&message[presize],
			    MAX_LOG_MSG_LENGTH - presize,
			    format, *varlist);
      
      message[MAX_LOG_MSG_LENGTH - 1] = '\0';
      
      if ( logp->log_print != NULL )
	{
           logp->log_print ((const char *) message);
	}
      else
	{
	  fprintf(stdout, "%s", message);
	}
    }
  
  return retvalue;
}  /* End of ms_log_main() */
