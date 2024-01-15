#include <stdio.h>
#include <math.h>

double f(int n, double *x, void *user_data) {
    double s = *(double *)user_data;
    return 1 / sqrt(2 * M_PI * s * s) / sqrt(2 * M_PI * x[0]) *
           exp(-0.5 * pow(x[1] - x[0], 2) / (s * s)) * exp(-0.5 * x[0]);
}

