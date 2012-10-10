/*--------------------------------------------------------------------
# Filename: stalta.c
#  Purpose: STA/LTA
#---------------------------------------------------------------------*/

#include <math.h>

typedef struct _headS {
    int N;
    int Nsta;
    int Nlta;
} headS;

int stalta(headS *head, double *data, double *charfct)
{
    int i, j;
    if (head->N < head->Nlta) {
        return 1;
    }

    for (i = 0; i < head->Nlta - 1; ++i) {
        charfct[i] = 0.0;
    }
    for (i = head->Nlta - 1; i < head->N; ++i) {
        double sta = 0.0;
        double lta;

        for (j = 0u; j < head->Nsta; j++) {
            sta += pow(data[i - j], 2);
        }
        lta = sta;
        for (j = head->Nsta; j < head->Nlta; j++) {
            lta += pow(data[i - j], 2);
        }
        sta /= (double) head->Nsta;
        lta /= (double) head->Nlta;
        charfct[i] = sta / lta;
    }
    return 0;
}
