#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "buf.h"
#include "gse_header.h"
#include "gse_types.h"
/******************************************************************
 *
 *	GSE2.0 example driver   (sts    14.4.2001)
 *
 ******************************************************************/
main()
{
    int i=0,ierr,ii=-1,i3=1;
    long data[20],*data2;
    int n,n2=0;
    long chksum=0,chksum2=0;       /* initialize checksum to zero! */
    FILE *fp, *fp2;
    struct header head, head2;
    char hline[121],tline[82]="";

    buf_init ();                   /* initialize the character buffer */
    fp = fopen ("test.gse","w");   /* open the output file */

    head.d_year=1998;              /* now, collect example */
    head.d_mon=4;                  /* header information */
    head.d_day=27;                 /* into the header structure */
    head.t_hour=9;                 /* */
    head.t_min=29;                 /* */
    head.t_sec=3.123;              /* */
    strcpy (head.station,"FELD "); /* */
    strcpy (head.channel,"SHZ");   /* */
    strcpy (head.auxid,"VEL ");    /* we have velocity data */
    strcpy (head.datatype,"CM6");  /* */
    head.n_samps=n=13;             /* */
    head.samp_rate=62.5;           /* */
    head.calib=1.;                 /* */
    head.calper=6.283;             /* that's ~ 2pi for our convenience */
    strcpy (head.instype,"LE-3D ");/* */
    head.hang=-1.0;                /* */
    head.vang=0.;                  /* header completed */

    printf(">>>> %4d/%02d/%02d %02d:%02d:%06.3f %-5s %-3s %-4s %-3s %8d %11.6f %10.4e %7.3f %-6s %5.1f %4.1f\n",
            head.d_year, head.d_mon, head.d_day, head.t_hour,
            head.t_min, head.t_sec, head.station, head.channel, head.auxid,
            head.datatype, head.n_samps, head.samp_rate, head.calib,
            head.calper, head.instype, head.hang, head.vang);

    data[0]=13;
    printf("%d %ld\n",i,data[0]);
    for (i=1;i<n;i++)              // let's create some data (13 points)
    {
        i3=i3*ii;
        data[i]=i3*data[i-1]*4+5;
        printf("%d %ld\n",i,data[i]);
    }

    /* 1st, compute the checksum */
    chksum=labs(check_sum (data,n,chksum));
    diff_2nd (data,n,0);           /* 2nd, compute the 2nd differences */
    ierr=compress_6b (data,n);     /* 3rd, character-encode the data */
    printf("error status after compression: %d\n",ierr);
    write_header(fp,&head);        /* 4th, write the header to the output file */
    buf_dump (fp);                 /* 5th, write the data to the output file */
    fprintf (fp,"CHK2 %8ld\n\n",chksum);  /* 6th, write checksum and closing line */
    fclose(fp);                    /* close the output file */
    buf_free ();                   /* clean up! */

    /* That's all! Now, we can try to read the GSE2.0 file */

    fp2 = fopen ("test.gse","r");
    read_header(fp2,&head2);       /* find and read the header line */

    printf("<<<< %4d/%02d/%02d %02d:%02d:%06.3f %-5s %-3s %-4s %-3s %8d %11.6f %10.4e %7.3f %-6s %5.1f %4.1f\n",
            head2.d_year, head2.d_mon, head2.d_day, head2.t_hour,
            head2.t_min, head2.t_sec, head2.station, head2.channel, head2.auxid,
            head2.datatype, head2.n_samps, head2.samp_rate, head2.calib,
            head2.calper, head2.instype, head2.hang, head2.vang);

    data2 = (long *) calloc (head2.n_samps,sizeof(long));   /* allocate data vector */
    n2 = decomp_6b (fp2, head2.n_samps, data2);   /* read and decode the data */
    printf("actual number of data read: %d\n",n2);
    rem_2nd_diff (data2, n2);      /* remove second differences */
    chksum=0;
    if (fgets(tline,82,fp2) == NULL)    /* read next line (there might be */
    { printf ("GSE: No CHK2 found before EOF\n"); /* an additional */
        return; }                                   /* blank line) */
        if (strncmp(tline,"CHK2",4))        /* and look for CHK2 */
        { if (fgets(tline,82,fp2) == NULL)   /* read another line */
            { printf ("GSE: No CHK2 found before EOF\n");
                return; } }
                if (strncmp(tline,"CHK2",4))
                { printf ("GSE: No CHK2 found!\n");
                    return; }
                    sscanf(tline,"%*s %ld",&chksum);           /* extract checksum */
                    chksum2 = check_sum (data2, n2, chksum2);  /* compute checksum from data */
                    printf("checksum read    : %ld \nchecksum computed: %ld\n",chksum,chksum2);
                    for (i=0;i<n2;i++)             /* print out data. We got velocity data */
                    {                              /* hence, we multiply by 2pi*calib/calper */
                        data2[i] = (long) data2[i] * 6.283 * head2.calib / head2.calper;
                        printf("%d %ld \n",i,data2[i]);
                    }
                    close(fp2);
                    free (data2);                  /* clean up */
}
