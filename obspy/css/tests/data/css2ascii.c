#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <time.h>

void print_usage(void)
{
    fprintf(stderr,"\nUsage:\n");
    fprintf(stderr,"css2ascii -i <inpfile> -o <outfile>\n");
    fprintf(stderr,"-i <inpfile>: Input file name\n");
    fprintf(stderr,"-o <outfile>: Output file basename.\n");
    fprintf(stderr,"[-h]: usage\n");
    return;
}

/* from db2ascii.cpp (FTAN Code) */
// T4 data format
float DecoderPcUnix(float inFloat){
    typedef union
    {
     float fal;
     int val;
     char ch[4];
    } unSet;
    unSet inSet;
    char wch[2];
    inSet.fal = inFloat;
    wch[0]= inSet.ch[0];
    wch[1]= inSet.ch[1];
    inSet.ch[0] = inSet.ch[3];
    inSet.ch[1] = inSet.ch[2];
    inSet.ch[2] = wch[1];
    inSet.ch[3] = wch[0];
    return inSet.fal;
}
// S4 data format
int DecoderPcUnixS4(int inInt){
    typedef union
    {
     int fal;
     int val;
     char ch[4];
    } unSet;
    unSet inSet;
    char wch[2];
    inSet.fal = inInt;
    wch[0]= inSet.ch[0];
    wch[1]= inSet.ch[1];
    inSet.ch[0] = inSet.ch[3];
    inSet.ch[1] = inSet.ch[2];
    inSet.ch[2] = wch[1];
    inSet.ch[3] = wch[0];
    return inSet.fal;
}


int main ( int argc, char *argv[] )
{
    int		n;
    FILE	*fpin,*fpout;
    char	ifname[128];
    char        ofname[128];
    char	tmp_name[128];
    float	*floatarray;
    float	floatval;
    int		*intarray;
    int		intval;
    int         c;
    extern char *optarg;        /* used for command line */
    long lSize;
    size_t result;

    if ( argc < 5 ) {
        print_usage();
	exit(-1);
    }

    while ( (c=getopt(argc,argv,"i:o:f:u:s:h")) != -1 ) {
        switch (c) {
            case 'i':
                strcpy(ifname,optarg);
                if ( (fpin=fopen(ifname,"rb")) == NULL ) {
                    fprintf(stderr,"could not open file %s for reading\n",ifname);
                    exit(-2);
                }
                break;
            case 'o':
                strcpy(ofname,optarg);
                break;
            case 'h':
                print_usage();
                exit(0);
                break;
        }
    }

    // obtain file size:
    fseek(fpin,0,SEEK_END);
    lSize = ftell(fpin)/sizeof(int);
    rewind(fpin);

    floatarray = (float *) malloc ((unsigned) (lSize* sizeof(float)));
    intarray = (int *) malloc ((unsigned) (lSize* sizeof(int)));

    // T4:
    //if ((result=fread(floatarray,sizeof(float),lSize,fpin)) != lSize ) {
    // S4:
    if ((result=fread(intarray,sizeof(int),lSize,fpin)) != lSize ) {
          fprintf (stderr,"Read %d samples, wanted %d samples.\n",(int)result,(int)lSize);
          exit(-2);
    }
    fprintf(stderr,"Found %d samples\n",(int)result);

    fclose(fpin);

    strcpy(tmp_name,"");
    strcpy(tmp_name,ofname);
    if ( (fpout=fopen(tmp_name,"w")) == NULL ) {
         fprintf(stderr,"could not open file %s for writing\n",tmp_name);
         exit(-3);
    }

    for (n=0;n<result;n++) {
            // T4:
            //floatval=floatarray[n];
            //floatval=DecoderPcUnix(floatval); // Swap byte order
            //fprintf(fpout,"%f\n",floatval);
            // S4:
            intval=intarray[n];
            intval=DecoderPcUnixS4(intval); // Swap byte order
            fprintf(fpout,"%d\n",intval);
    }

    fclose(fpout);
    free((void *)floatarray);
    free((void *)intarray);

    exit(0);
}
