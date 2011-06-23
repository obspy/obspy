#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <string.h>

#include <stdlib.h>

#include "./evresp.h"

/* error_exit:  prints a user supplied error message to stderr and exits with the user
                supplied error condition */

void error_exit(int cond, char *msg, ...) {
  va_list ap;
  char *p, *new_p, *sval, *prob;
  char fmt_str[MAXFLDLEN],sub_str[MAXFLDLEN];
  int ival, i;
  double dval;

  fprintf(stderr,"%s EVRESP ERROR: ", myLabel);
  va_start(ap, msg);
  for(p = msg; *p; p++) {
    if(*p != '%') {
      fprintf(stderr,"%c",*p);
    }
    else if(*p == '%') {
      sscanf(p,"%s\\",sub_str);
      if((prob = strchr((sub_str+1),'%')) != (char *)NULL) {
        *prob = '\0';
      }
      strncpy(fmt_str,sub_str,MAXFLDLEN);      /* just in case isn't followed by a format flag */
      for(i = strlen(sub_str)-1; i >= 0; i--) {
        if((prob = strchr("cdfges",*(sub_str+i))) == (char *)NULL)
          *(sub_str+i) = '\0';
        else
          break;
      }
      if(i > 0)                              /* then a format flag followed the '%' character */
        strncpy(fmt_str,sub_str,MAXFLDLEN);
      switch(*(fmt_str+strlen(fmt_str)-1)) {
      case 'c':
      case 'd':
        ival = va_arg(ap,int);
        fprintf(stderr,fmt_str,ival);
        break;
      case 'f':
      case 'g':
      case 'e':
        dval = va_arg(ap,double);
        fprintf(stderr,fmt_str,dval);
        break;
      case 's':
        sval = va_arg(ap,char *);
        fprintf(stderr,fmt_str,sval);
        break;
      default:
/*        fprintf(stderr,fmt_str); */
        break;
      }
      new_p = strstr(p,fmt_str);
      p = new_p + strlen(fmt_str) - 1;
    }
  }
  fprintf(stderr,"\n");
  fflush(stderr);
  exit(cond);
}

/* error_return:  prints a user supplied error message to stderr and returns control
                  to the calling routine at the point that that routine calls
                  'setjmp(jump_buffer)' */

void error_return(int cond, char *msg, ...) {
  va_list ap;
  char *p, *new_p, *sval, *prob;
  char fmt_str[MAXFLDLEN],sub_str[MAXFLDLEN];
  int ival, i;
  double dval;

  if(curr_file == (char *)NULL)
    curr_file = "<stdin>";

  if(GblChanPtr != NULL) {
    if(curr_seq_no >= 0)
      fprintf(stderr,"%s EVRESP ERROR (%s.%s.%s.%s [File: %s; Start date: %s; Stage: %d]):\n\t",
	    myLabel, GblChanPtr->staname, GblChanPtr->network, GblChanPtr->locid, GblChanPtr->chaname,
	    curr_file, GblChanPtr->beg_t, curr_seq_no);
    else if(strlen(GblChanPtr->staname))
      fprintf(stderr,"%s EVRESP ERROR (%s.%s.%s.%s [File: %s; Start date: %s]):\n\t",
	    myLabel, GblChanPtr->staname, GblChanPtr->network, GblChanPtr->locid, GblChanPtr->chaname,
	    curr_file,GblChanPtr->beg_t);
    else
      fprintf(stderr,"%s EVRESP ERROR [File: %s]):\n\t", myLabel, curr_file);
  }
  else
    fprintf(stderr,"%s EVRESP ERROR [File: %s]):\n\t", myLabel, curr_file);
  va_start(ap, msg);
  for(p = msg; *p; p++) {
    if(*p != '%') {
      fprintf(stderr,"%c",*p);
    }
    else if(*p == '%') {
      sscanf(p,"%s\\",sub_str);
      if((prob = strchr((sub_str+1),'%')) != (char *)NULL) {
        *prob = '\0';
      }
      strncpy(fmt_str,sub_str,MAXFLDLEN);      /* just in case isn't followed by a format flag */
      for(i = strlen(sub_str)-1; i >= 0; i--) {
        if((prob = strchr("cdfges",*(sub_str+i))) == (char *)NULL)
          *(sub_str+i) = '\0';
        else
          break;
      }
      if(i > 0)                              /* then a format flag followed the '%' character */
        strncpy(fmt_str,sub_str,MAXFLDLEN);
      switch(*(fmt_str+strlen(fmt_str)-1)) {
      case 'c':
      case 'd':
        ival = va_arg(ap,int);
        fprintf(stderr,fmt_str,ival);
        break;
      case 'f':
      case 'g':
      case 'e':
        dval = va_arg(ap,double);
        fprintf(stderr,fmt_str,dval);
        break;
      case 's':
        sval = va_arg(ap,char *);
        fprintf(stderr,fmt_str,sval);
        break;
      default:
/*        fprintf(stderr,fmt_str); */
        break;
      }
      new_p = strstr(p,fmt_str);
      p = new_p + strlen(fmt_str) - 1;
    }
  }
  fprintf(stderr,",\n\tskipping to next response now\n");
  fflush(stderr);
  longjmp(jump_buffer,cond);
}

