#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "symnmf.h"


size_t _getline(char** lineptr, size_t* n, FILE* stream) {
    /* Utility to read a line of characters from file, terminated by '\n' */
    char* buffer = NULL;
    size_t length = 0;
    int ch;

    if (!lineptr || !n || !stream) return 0;

    while ((ch = fgetc(stream)) != EOF) {
        char* new_buffer = realloc(buffer, length + 2);
        if (!new_buffer) {
            free(new_buffer);
            free(buffer);
            return 0;
        }
        buffer = new_buffer;

        buffer[length++] = (char)ch;
        if (ch == '\n') break;
    }

    if (length == 0 && ch == EOF) {
        free(buffer);
        return 0;
    }

    buffer[length] = '\0';

    *lineptr = buffer;
    *n = length + 1;
    return length;
}

int count_columns(const char* line) {
    /* Returns the number of columns in read matrix */
    const char* p;
    int count = 1;
    for (p = line; *p; ++p) {
        if (*p == ',') count++;
    }
    return count;
}

void compute_similarity(const double* X, int n, int d, double* A) {
    /* Computes similarity matrix A (n*n) from X (n*k)
    Assumes A is a pointer to a valid n*n array of doubles */
    int i;
    int j;
    int t;
    const double* xi;
    const double* xj;
    double dist2;
    for (i = 0; i < n; ++i) {
        xi = X + i * d;
        for (j = 0; j < n; ++j) {
            if (i == j) { A[i*n + j] = 0.0; continue; }
            xj = X + j * d;
            dist2 = 0.0;
            for (t = 0; t < d; ++t) {
                double diff = xi[t] - xj[t];
                dist2 += diff * diff;
            }
            A[i*n + j] = exp(-0.5 * dist2);
        }
    }
}

void compute_ddg(const double* A, int n, double* D) {
    /* Computes diagonal degree matrix D from similarity matrix A
    Assimes D is a valid pointer to an n*n array of doubles */
    int i;
    int j;
    for (i = 0; i < n*n; ++i) D[i] = 0.0;
    for (i = 0; i < n; ++i) {
        double s = 0.0;
        for (j = 0; j < n; ++j) s += A[i*n + j];
        D[i*n + i] = s;
    }
}

void compute_norm(const double* A, const double* D, int n, double* W) {
    /* Computes normalized similarity matrix W
    D: diagonal degree matrix, n*n
    A: similarity matrix, n*n
    Assumes valid pointers to n*n arrays of doubles */
    int i;
    int j;
    double* s = (double*)malloc((size_t)n * sizeof(double));
    for (i = 0; i < n; ++i) {
        double di = D[i*n + i];
        s[i] = di > 0.0 ? 1.0 / sqrt(di) : 0.0;
    }
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            W[i*n + j] = s[i] * A[i*n + j] * s[j];
        }
    }
    free(s);
}


static void matmul(const double* A, const double* B, double* C, int n, int m, int p) {
    /* Utility for matrix multiplication */
    int i;
    int j;
    int t;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < p; ++j) {
            double s = 0.0;
            for (t = 0; t < m; ++t) s += A[i*m + t] * B[t*p + j];
            C[i*p + j] = s;
        }
    }
}

void transpose(double* A, double* B, int n, int k) {
    /* Utility to transpose an n*k matrix A into k*n matrix B
    Assumes valid pointers to both A and B */
    int i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            B[j * n + i] = A[i * k + j];
        }
    }
}


static double frob_sq_diff(const double* A, const double* B, int n, int m) {
    /* Computes squared Frobenius norm */
    double s = 0.0;
    int i;
    for (i = 0; i < n*m; ++i) {
        double d = A[i] - B[i];
        s += d * d;
    }
    return s;
}

void symnmf_optimize(const double* W, int n, int k, int max_iter, double eps, double beta, double* H) {
    /* Runs symnmf optimization until convergence (until max_iter iterations or difference from previous < eps)*/
    double* WH   = (double*)malloc((size_t)n * k * sizeof(double));
    double* HHTH = (double*)malloc((size_t)n * k * sizeof(double));
    double* Hprev= (double*)malloc((size_t)n * k * sizeof(double));
    double* HHt  = (double*)malloc((size_t)n * n * sizeof(double));
    double* Ht  = (double*)malloc((size_t)n * n * sizeof(double));
    double ratio;
    double val;
    int it, i;

    for (it = 0; it < max_iter; it++) {
        memcpy(Hprev, H, (size_t)n*k*sizeof(double));
        transpose(H, Ht, n, k);               /* H^T (k x n) */
        matmul(W, H, WH, n, n, k);               /* WH */
        matmul(H, Ht, HHt, n, k, n);              /* H H^T (n x n) */
        matmul(HHt, H, HHTH, n, n, k);           /* (H H^T) H */

        for (i = 0; i < n*k; ++i) {
            double denom = HHTH[i];
            if (denom == 0.0) denom = 1e-12;
            ratio = WH[i] / denom;
            val = H[i] * (1.0 - beta + (beta * ratio));
            H[i] = val;
        }

        if (frob_sq_diff(H, Hprev, n, k) < eps) break;
    }

    free(WH); free(HHTH); free(Hprev); free(HHt);
}


void print_matrix(double* M, int n, int k) {
    /* Utility to pretty print an n*K matrix M */
    int i;
    int j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k - 1; j++) {
            printf("%.4f,", M[i * k + j]);
        }
        printf("%.4f\n", M[i * k + j]);
    }
}

void read_error(double* X, double* temp, FILE* fp, char* line) {
    /* To prevent code duplication while reading matrix */
    printf("An Error Has Occured\n");
    free(X);
    free(temp);
    free(line);
    fclose(fp);
    exit(1);
}

double* read_matrix(char* path, int* k, int* n) {
    /* Reads matrix from file, updates matrix dimentions n and k
    Assumes matrix is of doubles seperateed by commas */
    double* X = NULL;
    double* temp = NULL;
    int rows = 0;
    int cols = 0;
    int col;
    char* line = NULL;
    size_t len = 0;
    size_t read;
    char* token;
    FILE* fp = fopen(path, "r");
    if (!fp) {
        printf("An Error Has Occurred\n");
        exit(1);
    }
    while ((read = _getline(&line, &len, fp)) != 0) {
        line[strcspn(line, "\r\n")] = '\0';
        if (cols == 0) {
            cols = count_columns(line);
        }
        temp = realloc(X, (rows + 1) * cols * sizeof(double));
        if (!temp) {
            read_error(X, temp, fp, line);
        }
        X = temp;
        token = strtok(line, ",");
        for (col = 0; col < cols; col++) {
            if (!token) {
                read_error(X, temp, fp, line);
            }
            X[rows * cols + col] = strtod(token, NULL);
            token = strtok(NULL, ",");
        }
        rows++; free(line); line = NULL; len = 0;
    }
    fclose(fp);
    *n = rows; *k = cols;
    return X;
}

int main(int argc, char* argv[]) {
    int n;
    int d;
    char* goal = argv[1];
    char* path = argv[2];
    double* X = read_matrix(path, &d, &n);
    double* A;
    double* D;
    double* W;

    if((argc != 3) || (strcmp(goal, "ddg") && strcmp(goal, "norm") && strcmp(goal, "sym"))){
        free(X);
        printf("An Error Has Occured\n");
        exit(1);
    }

    A = malloc(sizeof(double) * n * n);
    compute_similarity(X, n, d, A);
    if(strcmp(goal, "sym") == 0){
        print_matrix(A, n, n);
    }

    D = malloc(sizeof(double) * n * n);
    compute_ddg(A, n, D);
    if(strcmp(goal, "ddg") == 0){
        print_matrix(D, n, n);
    }
    
    W = malloc(sizeof(double) * n * n);
    compute_norm(A, D, n, W);
    if (strcmp(goal, "norm") == 0){
        print_matrix(W, n, n);
    }
    free(X); free(A); free(D); free(W);
    return 0;
}