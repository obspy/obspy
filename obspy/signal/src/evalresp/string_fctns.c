/* string_fctns.c */

/*
   02/12/2005 -- [IGD] Moved parse_line() to ev_parse_line() to avoid name 
                       conflict	with external libraries
   10/21/2005 -- [ET]  Modified so as not to require characters after
                       'units' specifiers like "M" and "COUNTS";
                       improved error message generated when no
                       data fields found on line; added test of 'strstr()'
                       result in 'count_fields()' and 'parse_field()'
                       functions to prevent possible program crashes
                       due to null pointer (tended to be caused by
                       response files with Windows-type "CR/LF" line
                       ends); modified 'get/next/check_line()' functions
                       to make them strip trailing CR and LF characters
                       (instead of just LF character).
    1/18/2006 -- [ET]  Renamed 'regexp' functions to prevent name clashes
                       with other libraries.
     4/4/2006 -- [ET]  Modified 'parse_line()' and 'parse_delim_line()'
                       functions to return allocated string array with
                       empty entry (instead of NULL) if no fields found.
    8/21/2006 -- [IGD] Version 3.2.36: Added support for TESLA units
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>

#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif

#include <string.h>

#include "./evresp.h"
#include "./regexp.h"

/* ev_parse_line: parses the fields on a line into separate strings.  The definition of a field
               There is any non-white space characters with bordering white space.  The result
               is a structure containing the number of fields on the line and an array of
               character strings (which are easier to deal with than the original line).  A second
               argument (end_user_info) contains a string that is used to determine where to
               start parsing the line.  The character position immediately following the
               first occurrence of this string is used as the start of the line.  A null string
               can be used to indicate that the start of the line should be used. */

struct string_array *ev_parse_line(char *line) {
  char *lcl_line, field[MAXFLDLEN];
  int nfields, fld_len, i = 0;
  struct string_array* lcl_strings;

  lcl_line = line;
  nfields = count_fields(lcl_line);
  if(nfields > 0) {
    lcl_strings = alloc_string_array(nfields);
    for(i = 0; i < nfields; i++) {
      parse_field(line,i,field);
      fld_len = strlen(field) + 1;
      if((lcl_strings->strings[i] = (char *)malloc(fld_len*sizeof(char))) == (char *)NULL) {
        error_exit(OUT_OF_MEMORY,"ev_parse_line; malloc() failed for (char) vector");
      }
      strncpy(lcl_strings->strings[i],"",fld_len);
      strncpy(lcl_strings->strings[i],field,fld_len-1);
    }
  }
  else {      /* if no fields then alloc string array with empty entry */
    lcl_strings = alloc_string_array(1);
    if((lcl_strings->strings[0] = (char *)malloc(sizeof(char))) == (char *)NULL) {
      error_exit(OUT_OF_MEMORY,"ev_parse_line; malloc() failed for (char) vector");
    }
    strncpy(lcl_strings->strings[0],"",1);
  }
  return(lcl_strings);
}

/* parse_delim_line: parses the fields on a line into seperate strings.  The definition of a field
               There is any non-white space characters with bordering white space.  The result
               is a structure containing the number of fields on the line and an array of
               character strings (which are easier to deal with than the original line).  A second
               argument (end_user_info) contains a string that is used to determine where to
               start parsing the line.  The character position immediately following the
               first occurrence of this string is used as the start of the line.  A null string
               can be used to indicate that the start of the line should be used. */

struct string_array *parse_delim_line(char *line, char *delim) {
  char *lcl_line, field[MAXFLDLEN];
  int nfields, fld_len, i = 0;
  struct string_array* lcl_strings;

  lcl_line = line;
  nfields = count_delim_fields(lcl_line, delim);
  if(nfields > 0) {
    lcl_strings = alloc_string_array(nfields);
    for(i = 0; i < nfields; i++) {
      memset(field, 0, MAXFLDLEN);
      parse_delim_field(line,i,delim,field);
      fld_len = strlen(field) + 1;
      if((lcl_strings->strings[i] = (char *)malloc(fld_len*sizeof(char))) == (char *)NULL) {
        error_exit(OUT_OF_MEMORY,"parse_delim_line; malloc() failed for (char) vector");
      }
      strncpy(lcl_strings->strings[i],"",fld_len);
      strncpy(lcl_strings->strings[i],field,fld_len-1);
    }
  }
  else {      /* if no fields then alloc string array with empty entry */
    lcl_strings = alloc_string_array(1);
    if((lcl_strings->strings[0] = (char *)malloc(sizeof(char))) == (char *)NULL) {
      error_exit(OUT_OF_MEMORY,"parse_delim_line; malloc() failed for (char) vector");
    }
    strncpy(lcl_strings->strings[0],"",1);
  }
  return(lcl_strings);
}

/* get_field:  returns the indicated field from the next 'non-comment' line from a RESP file
               (return value is the length of the resulting field if successful, exits with
               error if no non-comment lines left in file or if expected blockette and field
               numbers do not match those found in the next non-comment line.
               Note:  here a field is any string of non-white characters surrounded by
                      white space */

int get_field(FILE *fptr, char *return_field, int blkt_no, int fld_no, char *sep,
              int fld_wanted) {
  char line[MAXLINELEN];

  /* first get the next non-comment line */

  get_line(fptr, line, blkt_no, fld_no, sep);

  /* then parse the field that the user wanted from the line get_line returned */

  parse_field(line, fld_wanted, return_field);

  /* and return the length of the field */

  return(strlen(return_field));
}

/* test_field:  returns the indicated field from the next 'non-comment' line from a RESP file
                (return value is the length of the resulting field if successful, returns with
                a value of zero if no non-comment lines left in file or if expected blockette
                and field numbers do not match those found in the next non-comment line.
                Note:  here a field is any string of non-white characters surrounded by
                       white space */

int test_field(FILE *fptr, char *return_field, int *blkt_no, int *fld_no, char *sep,
              int fld_wanted) {
  char line[MAXLINELEN];

  /* first get the next non-comment line */

  next_line(fptr, line, blkt_no, fld_no, sep);

  /* then parse the field that the user wanted from the line get_line returned */

  parse_field(line, fld_wanted, return_field);

  /* and return the length of the field */

  return(strlen(return_field));

}

/* get_line:  returns the next 'non-comment' line from a RESP file (return value is the
              length of the resulting line if successful, exits with error if no
              non-comment lines left in file or if expected blockette and field numbers
              do not match those found in the next non-comment line. */
/* SBH - 2004.079 added code to skip over valid lines that we didn't expect. 
   Support for SHAPE formatte RESP files, and to skip blank lines */

int get_line(FILE *fptr, char *return_line, int blkt_no, int fld_no, char *sep) {
  char *lcl_ptr, line[MAXLINELEN];
  char *res;
  int  lcl_blkt, lcl_fld, test;
  int tmpint;
  char tmpstr[200];
  size_t slen, i;

  test = fgetc(fptr);
  
  
  while(test != EOF && test == '#') {
    strncpy(line,"",MAXLINELEN-1);
    res = fgets(line, MAXLINELEN, fptr);
    if (res == NULL) {
      test = EOF;
      break;
    }
    test = fgetc(fptr);
  }


  if(test == EOF) {
    return(0);
  }
  else {
    ungetc(test,fptr);
    res = fgets(line, MAXLINELEN, fptr);
    if (res == NULL) {
      return 0;
    }

  slen = strlen(line);
  for (i = 0; i < slen; i++)
  {
    if ('\t' == line[i])
      line[i] = ' ';
  }


    /* check for blank line */
    tmpint = sscanf(line, "%s", tmpstr);
	
    if (tmpint == EOF) {
		return get_line(fptr, return_line, blkt_no, fld_no, sep);
    }

    tmpint = strlen(line);        /* strip any trailing CR or LF chars */
    while(tmpint > 0 && line[tmpint-1] < ' ')
      line[--tmpint] = '\0';
  }

  /*if(!line)
    error_return(UNEXPECTED_EOF, "get_line; no more non-comment lines found in file");*/

  test = parse_pref(&lcl_blkt, &lcl_fld, line);
  if(!test) {
    error_return(UNDEF_PREFIX,"get_line; unrecogn. prefix on the following line:\n\t  '%s'",line);
  }

  /* check the blockette and field numbers found on the line versus the expected values */

  if(blkt_no != lcl_blkt) {
    /* try to parse the next line */
    return get_line(fptr, return_line, blkt_no, fld_no, sep);
    /*
      removed by SBH 2004.079
      if(fld_no != lcl_fld) {
      error_return(PARSE_ERROR,"get_line; %s%s%3.3d%s%3.3d%s%2.2d%s%2.2d","blkt",
      " and fld numbers do not match expected values\n\tblkt_xpt=B",
      blkt_no, ", blkt_found=B", lcl_blkt, "; fld_xpt=F", fld_no,
      ", fld_found=F", lcl_fld);
      }
    */
  }
  else if(fld_no != lcl_fld) {
    /* try to parse the next line */
    return get_line(fptr, return_line, blkt_no, fld_no, sep);
    /*
      removed by SBH 2004.079
      error_return(PARSE_ERROR,"get_line (parsing blockette [%3.3d]); %s%2.2d%s%2.2d",
      lcl_blkt, "unexpected fld number\n\tfld_xpt=F", fld_no,
      ", fld_found=F", lcl_fld, lcl_blkt);
    */
  }

  if((lcl_ptr = strstr(line,sep)) == (char *)NULL) {
    error_return(UNDEF_SEPSTR, "get_line; seperator string not found"); 
  }
  else if((lcl_ptr - line) > (int)(strlen(line)-1)) {
    error_return(UNDEF_SEPSTR, "get_line; nothing to parse after seperator string"); 
  }

  lcl_ptr++;
  while(*lcl_ptr && isspace(*lcl_ptr)) {
    lcl_ptr++;
  }

  if((lcl_ptr - line) > (int)strlen(line)) {
    error_return(UNDEF_SEPSTR, "get_line; no non-white space after seperator string"); 
  }

  strncpy(return_line,lcl_ptr,MAXLINELEN);
  return(strlen(return_line));
}


/* next_line:  returns the next 'non-comment' line from a RESP file (return value is the
               fld_no of the resulting line if successful, returns a value of 0 if no
               non-comment lines left in file), regardless of the blockette and field
               numbers for the line (these values are returned as the values of the input
               pointer variables fld_no and blkt_no). */
/* SBH - 2004.079 added code to skip blank lines */

int next_line(FILE *fptr, char *return_line, int *blkt_no, int *fld_no, char *sep) {
  char *lcl_ptr, line[MAXLINELEN];
  char *res;
  int test;
  int tmpint;
  char tmpstr[200];

  test = fgetc(fptr);

  while(test != EOF && test == '#') {
    res = fgets(line, MAXLINELEN, fptr);
    if (res == NULL) {
      test = EOF;
      break;
    }
    test = fgetc(fptr);
  }

  if(test == EOF) {
    return(0);
  }
  else {
    ungetc(test,fptr);
    res = fgets(line, MAXLINELEN, fptr);
    if (res == NULL) {
      return 0;
    }
    tmpint = strlen(line);        /* strip any trailing CR or LF chars */
    while(tmpint > 0 && line[tmpint-1] < ' ')
      line[--tmpint] = '\0';
  }

  /* check for blank line */

  tmpint = sscanf(line, "%s", tmpstr);

  if (tmpint == EOF) {
	return   next_line(fptr, return_line, blkt_no, fld_no, sep);
  }

  test = parse_pref(blkt_no, fld_no, line);
  if(!test) {
    error_return(UNDEF_PREFIX,"get_field; unrecogn. prefix on the following line:\n\t  '%s'",line);
  }

  if((lcl_ptr = strstr(line,sep)) == (char *)NULL) {
    error_return(UNDEF_SEPSTR, "get_field; seperator string not found"); 
  }
  else if((lcl_ptr - line) > (int)(strlen(line)-1)) {
    error_return(UNDEF_SEPSTR, "get_field; nothing to parse after seperator string"); 
  }

  lcl_ptr++;
  while(*lcl_ptr && isspace(*lcl_ptr)) {
    lcl_ptr++;
  }  
  strncpy(return_line,lcl_ptr,MAXLINELEN);

  return(*fld_no);
}

/* count_fields:  counts the number of white space delimited fields on
                  a given input line */

int count_fields(char *line) {
  char *lcl_ptr, *new_ptr;
  char lcl_field[50];
  int nfields = 0, test;

  lcl_ptr = line;
         /* added test of 'strstr()' result -- 10/21/2005 -- [ET] */
  while(*lcl_ptr && (test=sscanf(lcl_ptr,"%s",lcl_field)) != 0 &&
                              (new_ptr=strstr(lcl_ptr,lcl_field)) != NULL) {
    lcl_ptr = new_ptr + strlen(lcl_field); 
    nfields++;
  }
  return(nfields);
}

/* count_delim_fields:  counts the number of fields delimited by the char "delim" on
                  a given input line (note: in this routine an empty string has one
                  field in it...with null length) */

int count_delim_fields(char *line, char *delim) {
  const char *lcl_ptr, *tmp_ptr;
  int nfields = 0;
  int line_len = 0;

  lcl_ptr = (const char *)line;
  while(*lcl_ptr && (tmp_ptr = strstr((lcl_ptr+line_len),delim)) != (char *)NULL) {
    line_len = (tmp_ptr - lcl_ptr + 1);
    nfields += 1;
  }
  if(strlen((lcl_ptr+line_len))) {
    nfields++;
  } else if (line_len && !strcmp((lcl_ptr+line_len-1),",")) {
    nfields++;
  }

  return(nfields);
}

/* parse_field:  returns a field from the input line (return value is the
                 length of the resulting field if successful, exits with error if no
                 field exists with that number */

int parse_field(char *line, int fld_no, char *return_field) {
  char *lcl_ptr, *new_ptr;
  char lcl_field[MAXFLDLEN];
  int nfields, i;

  nfields = count_fields(line);
  if(fld_no >= nfields) {
    if(nfields > 0) {
      error_return(PARSE_ERROR, "%s%d%s%d%s",
                 "parse_field; Input field number (",fld_no,
                 ") exceeds number of fields on line(",nfields,")");
    }
    else {
      error_return(PARSE_ERROR, "%s",
                              "parse_field; Data fields not found on line");
    }
  }

  lcl_ptr = line;
         /* added test of 'strstr()' result -- 10/21/2005 -- [ET] */
  for(i = 0; i < fld_no; i++) {
    sscanf(lcl_ptr,"%s",lcl_field);
    if((new_ptr=strstr(lcl_ptr,lcl_field)) == NULL)
      break;
    lcl_ptr = new_ptr + strlen(lcl_field);
  }

  sscanf(lcl_ptr,"%s",return_field);
  return(strlen(return_field));
}

/* parse_delim_field:  returns a field from the input line (return value is the
                 length of the resulting field if successful, exits with error if no
                 field exists with that number */

int parse_delim_field(char *line, int fld_no, char *delim, char *return_field) {
  char *lcl_ptr, *tmp_ptr = NULL;
  int nfields,  i;

  nfields = count_delim_fields(line, delim);
  if(fld_no >= nfields) {
    if(nfields > 0) {
      error_return(PARSE_ERROR, "%s%d%s%d%s",
                 "parse_delim_field; Input field number (",fld_no,
                 ") exceeds number of fields on line(",nfields,")");
    }
    else {
      error_return(PARSE_ERROR, "%s",
                        "parse_delim_field; Data fields not found on line");
    }
  }

  lcl_ptr = line;
  for(i = 0; i <= fld_no; i++) {
    tmp_ptr = strstr(lcl_ptr, delim);
    if(tmp_ptr && i < fld_no)
      lcl_ptr = tmp_ptr + 1;
  }

  if(tmp_ptr)
    strncpy(return_field, lcl_ptr, (tmp_ptr-lcl_ptr));
  else
    strncpy(return_field, lcl_ptr, strlen(lcl_ptr));

  return(strlen(return_field));
}

/* check_line:  returns the blockette and field numbers in the prefix of the next 'non-comment'
                line from a RESP file (return value 1 if a non-comment field is found
                or NULL if no non-comment line is found */
/* SBH - 2004.079 added code to skip blank lines */
int check_line(FILE *fptr, int *blkt_no, int *fld_no, char *in_line) {
  char  line[MAXLINELEN];
  char *res;
  int  test;
  char tmpstr[200];
  int tmpint;

  test = fgetc(fptr);
  while(test != EOF && test == '#') {
    res = fgets(line, MAXLINELEN, fptr);
    if (res == NULL) {
      test = EOF;
      break;
    }
    test = fgetc(fptr);
  }

/*
    while(test != EOF && (test == 10) {
    fgets(line, MAXLINELEN, fptr);
    test = fgetc(fptr);
  }
*/

  if(test == EOF) {
    return(0);
  }
  else {
    ungetc(test,fptr);
    res = fgets(line, MAXLINELEN, fptr);
    if (res == NULL) {
      return 0;
    }

    /* check for blank line */
	tmpint = sscanf(line, "%s", tmpstr);

	if (tmpint == EOF) {
		return check_line(fptr, blkt_no, fld_no, in_line);
	}

    tmpint = strlen(line);        /* strip any trailing CR or LF chars */
    while(tmpint > 0 && line[tmpint-1] < ' ')
      line[--tmpint] = '\0';
  }

  test = parse_pref(blkt_no, fld_no, line);
  if(!test) {
    error_return(UNDEF_PREFIX,"check_line; unrecogn. prefix on the following line:\n\t  '%s'",line);
  }

  strncpy(in_line,line,MAXLINELEN);
  return(1);
}

/* get_int:  uses get_fld to return the integer value of the input string.
             If the requested field is not a proper representation of a number, then
             'IMPROPER DATA TYPE' error is signaled */

int get_int(char *in_line) {
  int value;

  if(!is_int(in_line))
    error_return(IMPROP_DATA_TYPE,"get_int; '%s' is not an integer", in_line);
  value = atoi(in_line);
  return(value);
}

/* get_double:  uses get_fld to return the double-precision value of the input string.
                If the requested field is not a proper representation of a number, then
                'IMPROPER DATA TYPE' error is signaled */

double get_double(char *in_line) {
  double lcl_val;

  if(!is_real(in_line))
    error_return(IMPROP_DATA_TYPE ,"get_double; '%s' is not a real number",
                 in_line);
  lcl_val = atof(in_line);
  return(lcl_val);
}

/* check_units:  checks an incoming line for keys that indicate the units represented by a
                 filter.  If the units are not recognized, an 'UNDEFINED UNITS' error
                 condition is signaled.  If the user specified that 'default' unit should
                 be used, then the line is simply saved as the value of 'SEEDUNITS[DEFAULT]'
                 and no check of the units is made */

int check_units(char *line) {
  int i, first_flag = 0;

  if(!strlen(GblChanPtr->first_units)) {
    first_flag = 1;
    strncpy(GblChanPtr->first_units,line,MAXLINELEN);
    unitScaleFact = 1.0;             /* global variable used to change to MKS units */
  }
  else
    strncpy(GblChanPtr->last_units,line,MAXLINELEN);

  if(def_units_flag) {
    return(DEFAULT);
  }

  for(i = 0; i < (int)strlen(line); i++)
    line[i] = toupper(line[i]);

/* IGD 02/03/01 a restricted case of pessure data is added
 * We will play with string_match ater if more requests show
 * up for pressure data.
********************************************/
  if ((strncasecmp(line, "PA", 2) == 0) || (strncasecmp(line, "MBAR", 4) == 0))
	return(PRESSURE);


/* IGD 08/21/06 Added support for TESLA */
  if (strncasecmp(line, "T -", 3) == 0)
        return(TESLA);


  if(string_match(line,"^[CNM]?M/\\(?S\\*\\*2\\)?|^[CNM]?M/\\(?SEC\\*\\*2\\)?|M/S/S","-r")) {
    if(first_flag && !strncmp("NM",line,(size_t)2))
      unitScaleFact = 1.0e9;
    else if(first_flag && !strncmp("MM",line,(size_t)2))
      unitScaleFact = 1.0e3;
    else if(first_flag && !strncmp("CM",line,(size_t)2))
      unitScaleFact = 1.0e2;
    return(ACC);
  }
  else if(string_match(line,"^[CNM]?M/S|^[CNM]?M/SEC","-r")) {
    if(first_flag && !strncmp(line,"NM",2))
      unitScaleFact = 1.0e9;
    else if(first_flag && !strncmp(line,"MM",2))
      unitScaleFact = 1.0e3;
    else if(first_flag && !strncmp(line,"CM",2))
      unitScaleFact = 1.0e2;
    return(VEL);
  }
  else if(string_match(line,"^[CNM]?M[^A-Z/]?","-r")) {
    if(first_flag && !strncmp(line,"NM",2))
      unitScaleFact = 1.0e9;
    else if(first_flag && !strncmp(line,"MM",2))
      unitScaleFact = 1.0e3;
    else if(first_flag && !strncmp(line,"CM",2))
      unitScaleFact = 1.0e2;
    return(DIS);
  }
  else if(string_match(line,"^COUNTS?[^A-Z]?","-r") || string_match(line,"^DIGITAL[^A-Z]?","-r")) {
    return(COUNTS);
  }
  else if(string_match(line,"^V[^A-Z]?","-r") || string_match(line,"^VOLTS[^A-Z]?","-r")) {
    return(VOLTS);
  }
#ifdef LIB_MODE
  return (DEFAULT);
#else
  error_return(UNRECOG_UNITS, "check_units; units found ('%s') are not supported", line);
#endif
   return(0); /*We should not reach to here */
}

/* string_match:  compares an input string (string) with a regular espression
                  or glob-style "pattern" (expr) using the 're_comp()' and
                  're_exec()' functions (from stdlib.h).
                  -First, if the type-flag is set to the string "-g", the
                   glob-style 'expr' is changed so that any '*' characters
                   are converted to '.*' combinations and and '?' characters
                   are converted to '.' characters.
                  -If the type-flag is set to "-r" then no conversions are
                   necessary (the string is merely copied to the new location).
                  -Finally, the 'regexp_pattern' argument is passed through the 
                   're_comp()' routine (compiling the pattern), and the value of
                   're_exec(string)' is returned to the calling function */

int string_match(const char *string, char *expr, char *type_flag) {
  char lcl_string[MAXLINELEN], regexp_pattern[MAXLINELEN];
  int i = 0, glob_type, test;
  register regexp *prog;
  register char *lcl_ptr;


  memset(lcl_string, 0, sizeof(lcl_string));
  memset(regexp_pattern, 0, sizeof(regexp_pattern));
  strncpy(lcl_string, string, strlen(string));
  lcl_ptr = expr;
  if(!strcmp(type_flag,"-r"))
    glob_type = 0;
  else if(!strcmp(type_flag,"-g"))
    glob_type = 1;
  else {
    fprintf(stderr,"%s string_match; improper pattern type (%s)\n",myLabel, type_flag);
    fflush(stderr);
    exit(2);
  }
  while(*lcl_ptr && i < (MAXLINELEN-1)) {
    if(glob_type && *lcl_ptr == '?') {
      regexp_pattern[i++] = '.';
      lcl_ptr++;
    }
    else if(glob_type && *lcl_ptr == '*') {
      regexp_pattern[i++] = '.';
      regexp_pattern[i++] = '*';
      lcl_ptr++;
    }
    else
      regexp_pattern[i++] = *(lcl_ptr++);
  }
  regexp_pattern[i] = '\0';

  if((prog = evr_regcomp(regexp_pattern)) == NULL) {
    error_return(RE_COMP_FAILED,"string_match; pattern '%s' didn't compile", 
                 regexp_pattern);
  }
  lcl_ptr = lcl_string;
  test = evr_regexec(prog, lcl_ptr);

  free(prog);
  return(test);
}

/* is_int:  a function that tests whether a string can be converted into
            an integer using string_match() */

int is_int(const char *test) {
  char ipattern[MAXLINELEN];

  /* first check to see if is an integer prefixed by a plus or minus.  If not
     then check to see if is simply an integer */

  strncpy(ipattern,"^[-+]?[0-9]+$",MAXLINELEN);
  return(string_match(test,ipattern,"-r"));
}

/* is_real:  a function that tests whether a string can be converted into
             an double using string_match() */

int is_real(const char *test) {
  char fpattern[MAXLINELEN];
  strncpy(fpattern,"^[-+]?[0-9]+\\.?[0-9]*[Ee][-+]?[0-9]+$",MAXLINELEN);
  strcat(fpattern,"|^[-+]?[0-9]*\\.[0-9]+[Ee][-+]?[0-9]+$");
  strcat(fpattern,"|^[-+]?[0-9]+\\.?[0-9]*$");
  strcat(fpattern,"|^[-+]?[0-9]*\\.[0-9]+$");
  return(string_match(test,fpattern,"-r"));
}

/* is_time:  a function that tests whether a string looks like a time string
             using string_match() */

int is_time(const char *test) {
  char fpattern[MAXLINELEN];

  /* time strings must be in the format 'hh:mm:ss[.#####]', so more than 14
     characters is an error (too many digits) */

  if(is_int(test) && atoi(test) < 24)
    return(1);

  /* if gets this far, just check without the decimal, then with the decimal */

  strncpy(fpattern,"^[0-9][0-9]?:[0-9][0-9]$",MAXLINELEN);
  strcat(fpattern,"|^[0-9][0-9]?:[0-9][0-9]:[0-9][0-9]$");
  strcat(fpattern,"|^[0-9][0-9]?:[0-9][0-9]:[0-9][0-9]\\.[0-9]*$");
  return(string_match(test,fpattern,"-r"));
}

/* add_null:  add a null character to the end of a string
	where is a pointer to a character that specifies where
	the null character should be placed, the possible values
		
	are:
		'a'-> removes all of spaces, then adds null character
		'e'-> adds null character to end of character string
*/

int add_null(char *s, int len, char where) {
    int len_save;
    switch(where)
    {
    case 'a':        /* remove extra spaces from end of string */
        len_save = len;
        for( ; len >= 0; len--){    /* test in reverse order */
            if(!isspace(*(s+len))){
                if(*(s+len) == '\0'){
                    return(len);
                }
                else{
                    if(len != len_save)
                        len += 1;
                    *(s+len) = '\0';
                    return(len);
                }
            }
        }
        break;
    case 'e':        /* add null character to end of string */
        if(len > 0){
            *(s+len) = '\0';
            return(len);
        }
        break;
    }
    *s = '\0';
    return(0);
}

  int is_IIR_coeffs (FILE *fp,  int position)	{
   /* IGD Very narrow-specified function.                                    */     
   /* It is used to check out if we are using a FIR or IIR coefficients      */
   /* in the  blockette 54. This information is contained in the 10th field  */
   /* of this blockette (number of denominators)                             */
   /* if the number of denominators is 0: we got FIR. Otherwise, it is IIR   */
   /* is_IIR_coeff reads the text response file; finds the proper line and   */
   /* decides if the response is IIR or FIR                                  */
   /* The text file is then fseek to the original position and               */
   /*   the function returns:                                                */
   /* 1 if it is IIR; 0 if it is not IIR;                                    */
   /*  it returns 0  in case of the error, so use it with a caution!         */
   /* IGD I.Dricker ISTI i.dricker@isti.com 07/00 for evalresp 3.2.17        */
char line[500];
  int i, denoms, result;
  for (i=0; i<80; i++)	{/* enough to make sure we are getting field 10; not to much to get to the next blockette */
	result = fscanf(fp, "%s", line);
	if (result != 1)
		return 0;
  	if (strncmp (line, "B054F10", 7) == 0)
		break;
   }
  if (strncmp (line, "B054F10", 7) == 0)	{
	for (i=0; i<4; i++) {
		result = fscanf(fp, "%s", line);
	}
	if (result != 1)
		return 0;
	denoms = atoi (line);
	if (denoms == 0)
		result = 0;
	else
		result = 1;
  }
 else 
	result = 0;
  fseek (fp, position, SEEK_SET);
  return(result);
  }
