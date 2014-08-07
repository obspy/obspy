/*
 *	Copyright (c) 1991 by Columbia University.
 */

#ifndef __AH__
#define __AH__

/*
   ah.h --
   AH file format.
   Written by Dean Witte (6/11/85).
   Modified by Roger Davis (3/1/91).
   Updated by Irit Dolev (2/20/94).
*/

/* magic numbers (1100-1149 reserved for AH) */
#define	AH_VERSION_2_0		(1100)

/* data types */
#define	AH_UNDEFINED		((short) 0)
#define	AH_FLOAT		((short) 1)
#define	AH_COMPLEX		((short) 2)
#define AH_DOUBLE		((short) 3)

/* error codes */
#define	AH_SUCCESS		(0)		/* no error */
#define	AH_MISC			(-1)		/* miscellaneous error */
#define	AH_RDHEAD		(-2)		/* header read error */
#define	AH_WRHEAD		(-3)		/* header write error */
#define	AH_DATATYPE		(-4)		/* bad data type */
#define	AH_RDDATA		(-5)		/* data read error */
#define	AH_WRDATA		(-6)		/* data write error */
#define	AH_MEMALLOC		(-7)		/* memory allocation error */
#define	AH_BADARG		(-8)		/* invalid argument */
#define	AH_BADMATCH		(-9)		/* parameter mismatch */
#define AH_OPENFILE		(-10)		/* open file error */
#define AH_CALIBINFO		(-11)		/* no calibration info */

typedef	struct {
	float r;
	float i;
} complex;

typedef	struct {
	double r;
	double i;
} d_complex;

typedef struct {
	short yr;
	short mo;
	short day;
	short hr;
	short mn;
	float sec;
} ahtime;

typedef struct {
	char *code;		/* station code */
	char *chan;		/* lpz, spn, etc. */
	char *stype;		/* wwssn, hglp, etc. */
	char *recorder;		/* recorder serial number */
	char *sensor;		/* sensor serial number */
	float azimuth;		/* degrees east from north */
	float dip;              /* up = -90, down = +90 */
	double slat;		/* latitude */
	double slon;		/* longitude */
	float elev;		/* elevation */
	float DS;		/* gain	*/
	float A0;		/* normalization */
	short npoles;		/* calibration info */
	complex *poles;
	short nzeros;
	complex *zeros;
	char *scomment;		/* station comment */
} ahstation;

typedef struct {
	double lat;		/* latitude */
	double lon;		/* longitude */
	float depth;		/* depth */
	ahtime ot;		/* origin time */
	char *ecomment;		/* comment */
} ahevent;

typedef struct {
	short type;		/* data type */
	long ndata;		/* number of samples */
	float delta;		/* sampling interval in seconds */
	float maxamp;		/* maximum amplitude of record */
	ahtime abstime;		/* start time of record section */
	char *units;		/* data units */
	char *inunits;		/* input units of transfer function */
	char *outunits;		/* output units of transfer function */
	char *rcomment;		/* comment line */
	char *log;		/* log of data manipulations */
} ahrecord;

typedef struct {
	char *attr;
	char *val;
} ahattr;

typedef struct {
	int magic;
	unsigned long length;
	ahstation station;
	ahevent event;
	ahrecord record;
	short nusrattr;		/* user-defined information */
	ahattr *usrattr;
} ahhead;

#if defined(c_plusplus) || defined(__cplusplus)

extern "C" {
int		ah_appendstr(char **, char *);
int		ah_copyhead(ahhead *, ahhead *);
void		ah_discfft(complex *, int, int);
char *		ah_error(int);
void		ah_fftr(float *, int);
void		ah_fftri(float *, int);
void		ah_freedynamic(ahhead *);
void		ah_freemem(ahhead *, void *, XDR *, XDR *, FILE *);
ahattr *	ah_getattr(ahhead *, char *);
int		ah_getdata(ahhead *, void *, XDR *);
int		ah_gethead(ahhead *, XDR *);
int		ah_getrecord(ahhead *, void **, XDR *);
int		ah_maxamp(ahhead *, void *);
void *		ah_mkdataspace(ahhead *);
int		ah_nullhead(ahhead *);
int		ah_putdata(ahhead *, void *, XDR *);
int		ah_puthead(ahhead *, XDR *);
int		ah_putrecord(ahhead *, void *, XDR *);
void		ah_rdiscfft(float *, int, int);
int		ah_replacestr(char **, char *);
int		ah_seek(int, XDR *);
int		ah_typesz(ahhead *);
}

#else

extern int	ah_appendstr();
extern int	ah_copyhead();
extern void	ah_discfft();
extern char *	ah_error();
extern void	ah_fftr();
extern void	ah_fftri();
extern void	ah_freedynamic();
/* extern void	ah_freemem(); */
extern ahattr *	ah_getattr();
extern int	ah_getdata();
extern int	ah_gethead();
extern int	ah_getrecord();
extern int	ah_maxamp();
extern void *	ah_mkdataspace();
extern int	ah_nullhead();
extern int	ah_putdata();
extern int	ah_puthead();
extern int	ah_putrecord();
extern void	ah_rdiscfft();
extern int	ah_replacestr();
extern int	ah_seek();
extern int	ah_typesz();

#endif defined(c_plusplus) || defined(__cplusplus)

#endif __AH__
