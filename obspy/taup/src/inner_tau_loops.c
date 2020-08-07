/*--------------------------------------------------------------------
# Filename: inner_tau_loops.c
#  Purpose: Some critical parts of TauPy are written in C for performance
#           reasons.
#           each point.
#   Author: Lion Krischer
# Copyright (C) 2015 L. Krischer
#---------------------------------------------------------------------*/
#define _USE_MATH_DEFINES  // for Visual Studio
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Simple macros for easy array access.
// Be careful as these naturally apply to all function in this module.
#define RAY_PARAMS(I, J) ray_params[(I) * max_j + (J)]
#define TIME(I, J) time[(I) * max_j + (J)]
#define DIST(I, J) dist[(I) * max_j + (J)]
#define TIME_DIST(I, J) time_dist[(I) * 4 + (J)]
// Record array...change here if it changes on the Python side.
enum {
    TD_P,
    TD_TIME,
    TD_DIST,
    TD_DEPTH
};
#define LAYER(I, J) layer[(I) * 4 + (J)]
// Record array...change here if it changes on the Python side.
enum {
    SL_TOP_P,
    SL_TOP_DEPTH,
    SL_BOT_P,
    SL_BOT_DEPTH
};


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
void tau_branch_calc_time_dist_inner_loop(
    double *ray_params, double *time, double *dist,
    double *layer, double *time_dist, int max_i, int max_j,
    double max_ray_param, int allow_turn) {

    int i, j;
    double p, time_sum, dist_sum;

    for (i=0; i < max_i; i++) {
        p = RAY_PARAMS(i, 0);
        if (p > max_ray_param) {
            continue;
        }

        time_sum = 0.0;
        dist_sum = 0.0;

        for (j=0; j < max_j; j++) {
            if (p > LAYER(j, SL_TOP_P) || p > LAYER(j, SL_BOT_P)) {
                break;
            }
            time_sum += TIME(i, j);
            dist_sum += DIST(i, j);
        }

        /* Check if the last element includes a turn. */
        if (j < max_j && allow_turn) {
            if ((LAYER(j, SL_TOP_P) - p) * (p - LAYER(j, SL_BOT_P)) > 0) {
                time_sum += TIME(i, j);
                dist_sum += DIST(i, j);
            }
        }

        TIME_DIST(i, TD_TIME) = time_sum;
        TIME_DIST(i, TD_DIST) = dist_sum;
    }
    return;
}


int seismic_phase_calc_time_inner_loop(
    double degree,
    double max_distance,
    double *dist,
    double *ray_param,
    double *search_dist_results,
    int *ray_num_results,
    int count) {

    double temp_deg, rad_dist, search_dist;
    int n = 0;
    int ray_num = 0;
    int r = 0;

    temp_deg = fabs(degree);
    while (temp_deg > 360.0) {
        temp_deg -= 360.0;
    }
    if (temp_deg > 180.0) {
        temp_deg = 360.0 - temp_deg;
    }

    rad_dist = temp_deg * M_PI / 180.0;

    while (n * 2.0 * M_PI + rad_dist <= max_distance) {
        search_dist = n * 2 * M_PI + rad_dist;

        for (ray_num=0; ray_num < (count - 1); ray_num++) {
            if ((search_dist == dist[ray_num + 1]) &&
                ((ray_num + 1) != (count - 1))) {
                continue;
            }
            else if ((dist[ray_num] - search_dist) *
                     (search_dist - dist[ray_num + 1]) >= 0) {
                if ((ray_param[ray_num] == ray_param[ray_num + 1]) &&
                    count > 2) {
                    continue;
                }
                search_dist_results[r] = search_dist;
                ray_num_results[r] = ray_num;
                r += 1;
            }
        }

        search_dist = (n + 1) * 2.0 * M_PI - rad_dist;
        if (temp_deg != 180.0) {

            for (ray_num=0; ray_num < (count - 1); ray_num++) {
                if ((search_dist == dist[ray_num + 1]) &&
                    ((ray_num + 1) != (count - 1))) {
                    continue;
                }
                else if ((dist[ray_num] - search_dist) *
                         (search_dist - dist[ray_num + 1]) >= 0) {
                    if ((ray_param[ray_num] == ray_param[ray_num + 1]) &&
                        count > 2) {
                        continue;
                    }
                    search_dist_results[r] = search_dist;
                    ray_num_results[r] = ray_num;
                    r += 1;
                }
            }

        }
        n += 1;
    }
    return r;
}


void bullen_radial_slowness_inner_loop(
        double *layer,
        double *p,
        double *time,
        double *dist,
        double radius,
        int max_i) {
    int i;
    double B, sqrt_top, sqrt_bot;
    double arg_top, arg_bot, dumb_var;

    for (i=0; i<max_i; i++) {
        dumb_var = (LAYER(i, SL_BOT_DEPTH) - LAYER(i, SL_TOP_DEPTH));
        if ((dumb_var < 0.0000000001) || (dumb_var != dumb_var)) {continue;}

        B = log(LAYER(i, SL_TOP_P) / LAYER(i, SL_BOT_P)) /
            log((radius - LAYER(i, SL_TOP_DEPTH)) /
                (radius - LAYER(i, SL_BOT_DEPTH)));
        arg_top = pow(LAYER(i, SL_TOP_P), 2) - pow(p[i], 2);
        arg_bot = pow(LAYER(i, SL_BOT_P), 2) - pow(p[i], 2);
        if (arg_top < 0.) {arg_top = 0.;}
        if (arg_bot < 0.) {arg_bot = 0.;}
        sqrt_top = sqrt(arg_top);
        sqrt_bot = sqrt(arg_bot);

        dist[i] = (atan2(p[i], sqrt_bot) - atan2(p[i], sqrt_top)) / B;
        time[i] = (sqrt_top - sqrt_bot) / B;
    }
}
