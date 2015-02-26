
/*	structure for data file header	--	witte, 11 June 85	*/

#define AHHEADSIZE 1024
#define CODESIZE 6
#define CHANSIZE 6
#define STYPESIZE 8
#define COMSIZE 80
#define TYPEMIN 1
#define TYPEMAX 6
#define LOGSIZE 202
#define LOGENT 10
#define NEXTRAS 21
#define NOCALPTS 30

typedef	struct	{
		float	x;
		float	y;
		} vector;

/* these typedefs now in ah.h
typedef	struct	{
		float	r;
		float	i;
		} complex;

typedef struct  {
		double r;
		double i;
		} d_complex;
*/

typedef	struct	{
		float	xx;
		float	yy;
		float	xy;
		} tensor;

struct	ah_time	{
		short		yr;	/* year		*/
		short		mo;	/* month	*/
		short		day;	/* day		*/
		short		hr;	/* hour		*/
		short		mn;	/* minute	*/
		float		sec;	/* second	*/
		};

struct	calib	{
		complex		pole;	/* pole		*/
		complex		zero;	/* zero		*/
		};

struct	station_info	{
		char		code[CODESIZE];	/* station code		*/
		char		chan[CHANSIZE];	/* lpz,spn, etc.	*/
		char		stype[STYPESIZE];/* wwssn,hglp,etc.	*/
		float		slat;		/* station latitude 	*/
		float		slon;		/*    "    longitude 	*/
		float		elev;		/*    "    elevation 	*/
		float		DS;	/* gain	*/
		float		A0;	/* normalization */
		struct	calib	cal[NOCALPTS];	/* calibration info	*/
		};

struct	event_info	{
		float		lat;		/* event latitude	*/
		float		lon;		/*   "   longitude	*/
		float		dep;		/*   "   depth		*/
		struct	ah_time	ot;		/*   "   origin time 	*/
		char		ecomment[COMSIZE];	/*	comment line	*/
		};

struct	record_info	{
		short		type;	/* data type (int,float,...) 	*/
		long		ndata;	/* number of samples		*/
		float		delta;	/* sampling interval		*/
		float		maxamp;	/* maximum amplitude of record 	*/
		struct	ah_time	abstime;/* start time of record section */
		float		rmin;	/* minimum value of abscissa 	*/
		char		rcomment[COMSIZE];	/* comment line		*/
		char		log[LOGSIZE]; /* log of data manipulations */
		};

typedef struct {
		struct	station_info	station;	/* station info */
		struct	event_info	event;		/* event info	*/
		struct	record_info	record;		/* record info	*/
		float		extra[NEXTRAS];	/* freebies */
		} ahhed;


#define	FLOAT	1
#define	COMPLEX	2
#define	VECTOR	3
#define	TENSOR	4
#define	DOUBLE	6
