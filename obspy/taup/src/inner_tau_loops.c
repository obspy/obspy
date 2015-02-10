/*--------------------------------------------------------------------
# Filename: inner_tau_loops.c
#  Purpose: Some critical parts of TauPy are written in C for performance
#           reasons.
#           each point.
#   Author: Lion Krischer
# Copyright (C) 2015 L. Krischer
#---------------------------------------------------------------------*/
#include <math.h>
#include <stdbool.h>
#include <stdio.h>


#define PI 3.141592653589793

// Simple macros for easy array access.
// Be careful as these naturally apply to all function in this module.
#define RAY_PARAMS(I, J) ray_params[(I) * max_j + (J)]
#define MASK(I, J) mask[(I) * max_j + (J)]
#define MASK(I, J) mask[(I) * max_j + (J)]
#define TIME(I, J) time[(I) * max_j + (J)]
#define DIST(I, J) dist[(I) * max_j + (J)]
#define TIME_DIST(I, J) time_dist[(I) * 4 + (J)]
#define LAYER(I, J) layer[(I) * 4 + (J)]


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


/* Original Python version.
    # Degrees must be positive and between 0 and 180
    tempDeg = abs(degrees)
    # Don't just use modulo, as 180 would be equal to 0.
    while tempDeg > 360:
        tempDeg -= 360
    if tempDeg > 180:
        tempDeg = 360 - tempDeg
    radDist = tempDeg * math.pi / 180
    arrivals = []
    # Search all distances 2n*PI+radDist and 2(n+1)*PI-radDist that are
    # less than the maximum distance for this phase. This ensures that
    # we get the time for phases that accumulate more than 180 degrees
    # of distance, for instance PKKKKP might wrap all of the way around.
    # A special case exists at 180, so we skip the second case if
    # tempDeg==180.
    n = 0

    while n * 2 * math.pi + radDist <= self.maxDistance:
        # Look for arrivals that are radDist + 2nPi, i.e. rays that have
        # done more than n laps.
        searchDist = n * 2 * math.pi + radDist
        for rayNum in range(len(self.dist) - 1):
            if searchDist == self.dist[rayNum + 1] and \
                    rayNum + 1 != len(self.dist) - 1:
                # So we don't get 2 arrivals for the same ray.
                continue
            elif (self.dist[rayNum] - searchDist) * (
                    searchDist - self.dist[rayNum + 1]) >= 0:
                # Look for distances that bracket the search distance.
                if self.ray_param[rayNum] == self.ray_param[rayNum + 1] \
                        and len(self.ray_param) > 2:
                    # Here we have a shadow zone, so itis not really an
                    # arrival.
                    continue
                arrivals.append(self.linear_interp_arrival(
                    searchDist, rayNum, self.name, self.puristName,
                    self.source_depth))
        # Look for arrivals that are 2(n+1)Pi-radDist, i.e. rays that
        # have done more than one half lap plus some number of whole laps.
        searchDist = (n + 1) * 2 * math.pi - radDist
        if tempDeg != 180:
            for rayNum in range(len(self.dist) - 1):
                if searchDist == self.dist[rayNum + 1] \
                        and rayNum + 1 != len(self.dist) - 1:
                    # So we don't get 2 arrivals for the same ray.
                    continue
                elif (self.dist[rayNum] - searchDist) * (
                        searchDist - self.dist[rayNum + 1]) >= 0:
                    if self.ray_param[rayNum] == \
                            self.ray_param[rayNum + 1] \
                            and len(self.ray_param) > 2:
                        # Here we have a shadow zone, so it is not really
                        # an arrival.
                        continue
                    arrivals.append(self.linear_interp_arrival(
                        searchDist, rayNum, self.name, self.puristName,
                        self.source_depth))
        n += 1

    # Perhaps these are sorted by time in the java code?
    return arrivals
*/

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

    rad_dist = temp_deg * PI / 180.0;

    while (n * 2.0 * PI + rad_dist <= max_distance) {
        search_dist = n * 2 * PI + rad_dist;

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

        search_dist = (n + 1) * 2.0 * PI - rad_dist;
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

    for (i=0; i<max_i; i++) {
        // Record array...change here if it changes on the Python side.
        if ((LAYER(i, 3) - LAYER(i, 1)) < 0.0000000001) {
            continue;
        }

        B = log(LAYER(i, 0) / LAYER(i, 2)) /
            log((radius - LAYER(i, 1)) / (radius - LAYER(i, 3)));

        sqrt_top = sqrt(pow(LAYER(i, 0), 2.0) - pow(p[i], 2));
        sqrt_bot = sqrt(pow(LAYER(i, 2), 2.0) - pow(p[i], 2));

        dist[i] = (atan2(p[i], sqrt_bot) - atan2(p[i], sqrt_top)) / B;
        time[i] = (sqrt_top - sqrt_bot) / B;
    }
}