typedef signed int int32_t;
int32_t check_sum (int32_t *, int, int32_t);
void diff_2nd (int32_t *, int, int);
int compress_6b (int32_t *, int);
void write_header (FILE *, struct header *);
int read_header (FILE *, struct header *);
void rem_2nd_diff (int32_t *, int);
int decomp_6b (FILE *, int, int32_t *);
