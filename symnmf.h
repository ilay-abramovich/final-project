#ifndef SYMNMF_H
#define SYMNMF_H

void compute_similarity(const double* X, int n, int d, double* A);
void compute_ddg(const double* A, int n, double* D);
void compute_norm(const double* A, const double* D, int n, double* W);

void init_H_from_W_mean(const double* W, int n, int k, unsigned int seed, double* H);

void symnmf_optimize(const double* W, int n, int k,
                     int max_iter, double eps, double beta,
                     double* H);

#endif
