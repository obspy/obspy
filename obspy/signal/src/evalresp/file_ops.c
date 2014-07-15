#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

/* file_ops.c

    8/28/2001 -- [ET]  Added 'WIN32' directives for Windows compiler
                       compatibility; added code to use 'findfirst()' and
                       'findnext()' (instead of 'ls') when using a Windows
                       compiler.
   10/21/2005 -- [ET]  Modified 'get_names()' function to work with
                       Microsoft compiler (under Windows).
    3/28/2006 -- [ET]  Fixed description of "mode" in header comment for
                       'find_files()' function (changed second "zero" to
                       "one").
    2/25/2010 -- [CT]  Convert 'get_names()' used under non-Windows to use
                       to use the system glob() to find files instead of 
		       forking a child to run 'ls' in a sub-process.

 */

#include <sys/types.h>
#include <sys/stat.h>

#include <stdlib.h>
#ifndef WIN32                /* if not Windows compiler then */
#include <sys/param.h>       /* include header files */
#include <unistd.h>
#include <glob.h>
#include <errno.h>
#include <sys/time.h>
#else                        /* if Windows compiler then */
#include <time.h>            /* 'time.h' is not in 'sys/' */
#endif
#include <string.h>
#include "evresp.h"

#ifdef WIN32
#if __BORLANDC__        /* if Borland compiler then */
#include <dir.h>        /* include header file for directory functions */
#else                   /* if non-Borland (MS) compiler then */
#include <io.h>         /* include header files for directory functions */
#include <direct.h>          /* define macro used below: */
#define S_ISDIR(m) ((m) & S_IFDIR)
#endif
#endif

/* find_files:

   creates a linked list of files to search based on the filename and
   scn_lst input arguments, i.e. based on the filename (if it is non-NULL)
   and the list of stations and channels.

   If the filename exists as a directory, then that directory is searched
   for a list of files that look like 'RESP.NETCODE.STA.CHA'.  The names of
   any matching files will be placed in the linked list of file names to
   search for matching response information.  If no match is found for a
   requested 'RESP.NETCODE.STA.CHA' file in that directory, then the search
   for a matching file will stop (see discussion of SEEDRESP case below).

   If the filename is a file in the current directory, then a (matched_files *)NULL
   will be returned

   if the filename is NULL the current directory and the directory indicated
   by the environment variable 'SEEDRESP' will be searched for files of
   the form 'RESP.NETCODE.STA.CHA'.  Files in the current directory will
   override matching files in the directory pointed to by 'SEEDRESP'.  The
   routine will behave exactly as if the filenames contained in these two
   directories had been specified

   the mode is set to zero if the user named a specific filename and
   to one if the user named a directory containing RESP files (or if the
   SEEDRESP environment variable was used to find RESP files

   if a pattern cannot be found, then a value of NULL is set for the
   'names' pointer of the linked list element representing that station-
   channel-network pattern.

   */

struct matched_files *find_files(char *file, struct scn_list *scn_lst, int *mode) {
  char *basedir, testdir[MAXLINELEN];
  char *tmpdir;
  char comp_name[MAXLINELEN], new_name[MAXLINELEN];
  int i, nscn, nfiles;
  struct matched_files *flst_head, *flst_ptr, *tmp_ptr;
  struct scn *scn_ptr;
  struct stat buf;

  /* first determine the number of station-channel-networks to look at */

  nscn = scn_lst->nscn;

  /* allocate space for the first element of the file pointer linked list */

  flst_head = alloc_matched_files();

  /* and set an 'iterator' variable to be moved through the linked list */

  flst_ptr = flst_head;

  /* set the value of the mode to 1 (indicating that a filename was
     not specified or a directory was specified) */

  *mode = 1;

  /* if a filename was given, check to see if is a directory name, if not
     treat it as a filename */

  if (file != NULL && strlen(file) != 0) {
    stat(file,&buf);
    if(S_ISDIR(buf.st_mode)){
      for(i = 0; i < nscn; i++) {
        scn_ptr = scn_lst->scn_vec[i];
        memset(comp_name,0,MAXLINELEN);
        sprintf(comp_name, "%s/RESP.%s.%s.%s.%s",file,
                scn_ptr->network,scn_ptr->station,scn_ptr->locid,scn_ptr->channel);
        nfiles = get_names(comp_name,flst_ptr);
        if(!nfiles && strcmp(scn_ptr->locid,"*")) {

          fprintf(stderr,"WARNING: evresp_; no files match '%s'\n",comp_name);
          fflush(stderr);
        }
        else if(!nfiles && !strcmp(scn_ptr->locid,"*")) {
          memset(comp_name,0,MAXLINELEN);
          sprintf(comp_name, "%s/RESP.%s.%s.%s",file,
                  scn_ptr->network,scn_ptr->station,scn_ptr->channel);
          nfiles = get_names(comp_name,flst_ptr);
          if(!nfiles) {
            fprintf(stderr,"WARNING: evresp_; no files match '%s'\n",comp_name);
            fflush(stderr);
          }
        }
        tmp_ptr = alloc_matched_files();
        flst_ptr->ptr_next = tmp_ptr;
        flst_ptr = tmp_ptr;
      }
    }
    else     /* file was specified and is not a directory, treat as filename */
      *mode = 0;
  }
  else {
    for(i = 0; i < nscn; i++) {      /* for each station-channel-net in list */
      scn_ptr = scn_lst->scn_vec[i];
      memset(comp_name,0,MAXLINELEN);
      sprintf(comp_name, "./RESP.%s.%s.%s.%s",
              scn_ptr->network,scn_ptr->station,scn_ptr->locid,scn_ptr->channel);
      if ((basedir = (char *) getenv("SEEDRESP")) != NULL) {
        /* if the current directory is not the same as the SEEDRESP
           directory (and the SEEDRESP directory exists) add it to the
           search path */
        stat(basedir, &buf);
        tmpdir = getcwd(testdir, MAXLINELEN);
        if(tmpdir && S_ISDIR(buf.st_mode) && strcmp(testdir, basedir)) {
	  memset(new_name,0,MAXLINELEN);
          sprintf(new_name, " %s/RESP.%s.%s.%s.%s",basedir,
                  scn_ptr->network,scn_ptr->station,scn_ptr->locid,scn_ptr->channel);
          strcat(comp_name,new_name);
        }
      }
      nfiles = get_names(comp_name,flst_ptr);
      if(!nfiles && strcmp(scn_ptr->locid,"*")) {
        fprintf(stderr,"WARNING: evresp_; no files match '%s'\n",comp_name);
        fflush(stderr);
      }
      else if(!nfiles && !strcmp(scn_ptr->locid,"*")) {
	memset(comp_name,0,MAXLINELEN);
        sprintf(comp_name, "./RESP.%s.%s.%s",scn_ptr->network,scn_ptr->station,
                scn_ptr->channel);
        if(basedir != NULL) {
          stat(basedir, &buf);
          tmpdir = getcwd(testdir, MAXLINELEN);
          if(tmpdir && S_ISDIR(buf.st_mode) && strcmp(testdir, basedir)) {
	    memset(new_name,0,MAXLINELEN);
            sprintf(new_name, " %s/RESP.%s.%s.%s",basedir,
                    scn_ptr->network,scn_ptr->station,scn_ptr->channel);
            strcat(comp_name,new_name);
          }
        }
        nfiles = get_names(comp_name,flst_ptr);
        if(!nfiles) {
          fprintf(stderr,"WARNING: evresp_; no files match '%s'\n",comp_name);
          fflush(stderr);
        }
      }
      tmp_ptr = alloc_matched_files();
      flst_ptr->ptr_next = tmp_ptr;
      flst_ptr = tmp_ptr;
    }
  }

  /* return the pointer to the head of the linked list, which is null
     if no files were found that match request */

  return(flst_head);

}

#ifndef WIN32      /* if not Windows then use original 'get_names()' */

/* get_names:  uses system glob() to get filenames matching the
               expression in 'in_file'. */

int get_names(char *in_file, struct matched_files *files) {
  struct file_list *lst_ptr, *tmp_ptr;
  glob_t globs;
  int count;
  int rv;
  
  /* Search for matching file names */
  if ( (rv = glob (in_file, 0, NULL, &globs)) ) {
    if ( rv != GLOB_NOMATCH )
      perror("glob");
    return 0;
  }
  
  /* set the head of the 'files' linked list to a pointer to a newly allocated
     'matched_files' structure */
  
  files->first_list = alloc_file_list();
  tmp_ptr = lst_ptr = files->first_list;
  
  /* retrieve the files from the glob list and build up a linked
     list of matching files */
  
  count = globs.gl_pathc;
  while( count ) {
    count--;
    files->nfiles++;
    lst_ptr->name = alloc_char(strlen(globs.gl_pathv[count])+1);
    strcpy(lst_ptr->name,globs.gl_pathv[count]);
    lst_ptr->next_file = alloc_file_list();
    tmp_ptr = lst_ptr;
    lst_ptr = lst_ptr->next_file;
  }
  
  /* allocated one too many files in the linked list */
  
  if(lst_ptr != (struct file_list *)NULL) {
    free_file_list(lst_ptr);
    free(lst_ptr);
    if(tmp_ptr != lst_ptr)
      tmp_ptr->next_file = (struct file_list *)NULL;
  }
  
  globfree (&globs);
  
  return(files->nfiles);
}

#else              /* if Windows compiler then use new 'get_names()' */

/* get_names:  uses 'findfirst()' and 'findnext()' to get filenames
               matching the expression in 'in_file'. */

int get_names(char *in_file, struct matched_files *files)
{
  struct file_list *lst_ptr, *tmp_ptr;
#if __BORLANDC__             /* if Borland compiler then */
  struct ffblk fblk;         /* define block for 'findfirst()' fn */
#define findclose()
#else                        /* if non-Borland (MS) compiler then */
  struct _finddata_t fblk;   /* define block for 'findfirst()' fn */
         /* setup things for Microsoft compiler compatibility: */
int fhandval;
#define ff_name name
#define findfirst(name,blk,attrib) (fhandval=_findfirst(name,blk))
#define findnext(blk) _findnext(fhandval,blk)
#define findclose() _findclose(fhandval)
#endif

  if(findfirst(in_file,&fblk,0) < 0)
  {      /* no matching files found */
    findclose();        /* release resources for findfirst/findnext */
    return 0;
  }

  files->first_list = alloc_file_list();
  lst_ptr = files->first_list;

  /* retrieve the files and build up a linked
     list of matching files */
  do
  {
    files->nfiles++;
    lst_ptr->name = alloc_char(strlen(fblk.ff_name)+1);
    strcpy(lst_ptr->name,fblk.ff_name);
    lst_ptr->next_file = alloc_file_list();
    tmp_ptr = lst_ptr;
    lst_ptr = lst_ptr->next_file;
  }
  while(findnext(&fblk) >= 0);
  findclose();          /* release resources for findfirst/findnext */

  /* allocated one too many files in the linked list */
  if(lst_ptr != (struct file_list *)NULL) {
    free_file_list(lst_ptr);
    free(lst_ptr);
    if(tmp_ptr != lst_ptr)
      tmp_ptr->next_file = (struct file_list *)NULL;
  }

  return(files->nfiles);
}

#endif

