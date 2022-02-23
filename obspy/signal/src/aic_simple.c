/**
 * Simple AIC (Akaike information criterion) by Maeda (1985).
 * Author: Danylo Ulianych
 * Copyright (C) 2022 D. Ulianych
 */

#include <stdint.h>
#include <stdlib.h>
#include <math.h>


typedef struct {
	double mean;
	double varsum;  // variance sum
	uint32_t count;
} OnlineMean;


void OnlineMean_Init(OnlineMean *oMean, double firstVal) {
    oMean->mean = firstVal;
    oMean->varsum = 0;
    oMean->count = 1;
}

/**
 * Welford's online algorithm of estimation the population mean & variance.
 */
void OnlineMean_Update(OnlineMean *oMean, double newVal) {
    oMean->count++;
    double delta = newVal - oMean->mean;
    oMean->mean += delta / oMean->count;
    oMean->varsum += delta * (newVal - oMean->mean);
}


void aic_simple(double *aic, const double *arr, uint32_t size) {
    OnlineMean oMean;

    if (size <= 2) {
        for (int i = 0; i < size; i++) {
            aic[i] = 0;
        }
        return;
    }

    // special case: the forward loop skips the first element; init with zero
    aic[0] = 0;

    OnlineMean_Init(&oMean, arr[0]);
    for (uint32_t i = 1; i < size - 1; i++) {
        // forward loop: AIC(i) = (i + 1) log(Var{0..i})
        OnlineMean_Update(&oMean, arr[i]);
        aic[i] = oMean.count * log(oMean.varsum / oMean.count);
    }

    OnlineMean_Init(&oMean, arr[size - 1]);
    for (uint32_t i = size - 2; i > 0; i--) {
        // backward loop: AIC(i) += (N - i - 2) log(Var{i+1..N-1})
        // or AIC(i-1) += (N - i - 1) log(Var{i..N-1})
        OnlineMean_Update(&oMean, arr[i]);
        aic[i - 1] += (oMean.count - 1) * log(oMean.varsum / oMean.count);
    }

    // append the last element
    aic[size - 1] = aic[size - 2];
}
