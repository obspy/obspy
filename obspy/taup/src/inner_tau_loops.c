/*--------------------------------------------------------------------
# Filename: inner_tau_loops.c
#  Purpose: Some critical parts of TauPy are written in C for performance
#           reasons.
#           each point.
#   Author: Lion Krischer
# Copyright (C) 2015 L. Krischer
#---------------------------------------------------------------------*/

/*
Original Python version:

for i, p in enumerate(ray_params[:, 0]):
    if p <= self.maxRayParam:
        m = mask[i]
        layerMask = layer[m]
        if len(layerMask):
            timeDist['time'][i] = np.sum(time[i][m])
            timeDist['dist'][i] = np.sum(dist[i][m])

            # This check is not part of the C version as it is never
            # executed so I think it is not needed.
            if ((layerMask['topP'][-1] - p) *
                    (p - layerMask['botP'][-1])) > 0:
                raise SlownessModelError(
                    "Ray turns in the middle of this layer!")
*/

#include <stdbool.h>

// Simple macros for easy array access.
#define RAY_PARAMS(I, J) ray_params[(I) * max_j + (J)]
#define MASK(I, J) mask[(I) * max_j + (J)]
#define MASK(I, J) mask[(I) * max_j + (J)]
#define TIME(I, J) time[(I) * max_j + (J)]
#define DIST(I, J) dist[(I) * max_j + (J)]
#define TIME_DIST(I, J) time_dist[(I) * 4 + (J)]

void tau_branch_calc_time_dist_inner_loop(
    double *ray_params, bool *mask, double *time, double *dist,
    double *layer, double *time_dist, int max_i, int max_j,
    double max_ray_param) {

    int i, j;
    bool m;
    double p, time_sum, dist_sum;

    for (i=0; i < max_i; i++) {
        p = RAY_PARAMS(i, 0);
        if (p > max_ray_param) {
            continue;
        }

        time_sum = 0.0;
        dist_sum = 0.0;

        for (j=0; j < max_j; j++) {
            m = MASK(i, j);
            if (!m) {
                continue;
            }
            time_sum += TIME(i, j);
            dist_sum += DIST(i, j);
        }

        // Record array. Change here if record array on the Python side
        // changes!
        TIME_DIST(i, 1) = time_sum;
        TIME_DIST(i, 2) = dist_sum;
    }
    return;
}