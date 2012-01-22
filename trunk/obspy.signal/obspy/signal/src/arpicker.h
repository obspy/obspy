#ifndef ARPICK
#define ARPICK

int ar_picker(float *tr, float *tr_1, float *tr_2, int   ndat, float sample_rate, float f1, float f2, float lta_p, float sta_p, float lta_s, float sta_s, int m_p, int m_s, float *ptime, float *stime, double l_p, double l_s, int s_pick);

void spr_bp_fast_bworth(float *tr, int ndat, float tsa, float flo, float fhi, int ns, int zph);

int spr_coef_paz(float *data,int n,int m, /*@out@*/ float *fp, /*@out@*/ float *coef);
#endif

#define TRUE 1
#define FALSE 0

