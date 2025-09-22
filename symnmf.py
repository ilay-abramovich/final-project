
import sys
import numpy as np
import symnmfmodule as sm  # required
from consts import *

def print_matrix(M):
    """
    Utility function to print a matrix with 4 decimal places.
    """
    rows = []
    for r in M:
        rows.append(",".join(f"{x:.4f}" for x in r))
    print("\n".join(rows))

def init_H(W, k):
    """
    Initialize H0 from W mean as described in the assignment.
    """
    np.random.seed(SEED)
    n = W.shape[0]
    m = W.mean()
    upper = 2.0 * np.sqrt(m / max(k, 1))
    H0 = np.random.uniform(0.0, upper, size=(n, k))
    return H0

def read_points(path):
    try:
        X = []
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                X.append([float(v) for v in s.split(",")])
        return np.asarray(X, dtype=np.float64)
    except Exception:
        print(ERROR_MESSAGE)
        sys.exit(1)

def run_symnmf(X, k):
    """
    Wrapper for symnmf function from the C extension.
    """
    W = sm.norm(X)
    H0 = init_H(W, k)
    H = sm.symnmf(H0, W, k, EPS, DEFAULT_ITER, BETA)
    return H

def main():
    if len(sys.argv) != 4:
        print(ERROR_MESSAGE)
        sys.exit(1)
    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        file_name = sys.argv[3]
        X = read_points(file_name)
        match goal:
            case "sym":
                A = sm.sym(X)
                print_matrix(A)

            case "ddg":
                D = sm.ddg(X)
                print_matrix(D)

            case "norm":
                W = sm.norm(X)
                print_matrix(W)

            case "symnmf":
                H = run_symnmf(X, k)
                print_matrix(H)

            case _:
                print(ERROR_MESSAGE)
                sys.exit(1)

    except Exception as e:
        print(ERROR_MESSAGE)
        sys.exit(1)

if __name__ == "__main__":
    main()
