#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "symnmf.h"

static inline double dot(const double* a, const double* b, int d) {
    double s = 0.0;
    for (int i = 0; i < d; ++i) s += a[i] * b[i];
    return s;
}

void compute_similarity(const double* X, int n, int d, double* A) {
    for (int i = 0; i < n; ++i) {
        const double* xi = X + i * d;
        for (int j = 0; j < n; ++j) {
            if (i == j) { A[i*n + j] = 0.0; continue; }
            const double* xj = X + j * d;
            double dist2 = 0.0;
            for (int t = 0; t < d; ++t) {
                double diff = xi[t] - xj[t];
                dist2 += diff * diff;
            }
            A[i*n + j] = exp(-0.5 * dist2);
        }
    }
}

void compute_ddg(const double* A, int n, double* D) {
    for (int i = 0; i < n*n; ++i) D[i] = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += A[i*n + j];
        D[i*n + i] = s;
    }
}

void compute_norm(const double* A, const double* D, int n, double* W) {
    double* s = (double*)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        double di = D[i*n + i];
        s[i] = di > 0.0 ? 1.0 / sqrt(di) : 0.0;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            W[i*n + j] = s[i] * A[i*n + j] * s[j];
        }
    }
    free(s);
}

void init_H_from_W_mean(const double* W, int n, int k, unsigned int seed, double* H) {
    double mean = 0.0;
    for (int i = 0; i < n*n; ++i) mean += W[i];
    mean /= (double)(n*n);
    double upper = 2.0 * sqrt(mean / (k > 0 ? k : 1));
    unsigned int state = seed ? seed : 1234u;
    for (int i = 0; i < n*k; ++i) {
        state = 1664525u * state + 1013904223u;
        double u = (state / (double)UINT_MAX);
        H[i] = u * upper;
        if (H[i] < 0.0) H[i] = 0.0;
    }
}

static void matmul(const double* A, const double* B, double* C, int n, int m, int p) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            double s = 0.0;
            for (int t = 0; t < m; ++t) s += A[i*m + t] * B[t*p + j];
            C[i*p + j] = s;
        }
    }
}

static double frob_sq_diff(const double* A, const double* B, int n, int m) {
    double s = 0.0;
    for (int i = 0; i < n*m; ++i) {
        double d = A[i] - B[i];
        s += d * d;
    }
    return s;
}

void symnmf_optimize(const double* W, int n, int k,
                     int max_iter, double eps, double beta,
                     double* H) {
    double* WH   = (double*)malloc((size_t)n * k * sizeof(double));
    double* HHTH = (double*)malloc((size_t)n * k * sizeof(double));
    double* Hprev= (double*)malloc((size_t)n * k * sizeof(double));
    for (int it = 0; it < max_iter; ++it) {
        memcpy(Hprev, H, (size_t)n*k*sizeof(double));

        matmul(W, H, WH, n, n, k);               /* WH */
        double* HHt = (double*)malloc((size_t)n * n * sizeof(double));
        matmul(H, H, HHt, n, k, n);              /* H H^T (n x n) */
        matmul(HHt, H, HHTH, n, n, k);           /* (H H^T) H */
        free(HHt);

        for (int i = 0; i < n*k; ++i) {
            double denom = HHTH[i];
            if (denom == 0.0) denom = 1e-12;
            double ratio = WH[i] / denom;
            double val = H[i] * (1.0 - beta + beta * ratio);
            H[i] = (val > 0.0) ? val : 0.0;
        }

        if (frob_sq_diff(H, Hprev, n, k) < eps) break;
    }
    free(WH); free(HHTH); free(Hprev);
}
