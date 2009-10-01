long check_sum (long *, int, long);
void diff_2nd (long *, int, int);
int compress_6b (long *, int);
void write_header (FILE *, struct header *);
int read_header (FILE *, struct header *);
void rem_2nd_diff (long *, int);
int decomp_6b (FILE *, int, long *);
