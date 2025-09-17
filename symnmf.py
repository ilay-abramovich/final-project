
# symnmf.py — Python CLI that calls the C extension (per spec)

import sys
import numpy as np
import symnmfmodule as sm  # required
EPS = 1e-4
MAX_ITER = 300
BETA = 0.5


def init_H_from_W_mean(W: np.ndarray, k: int, seed: int = 1234) -> np.ndarray:
    m = float(W.mean())
    upper = 2.0 * np.sqrt(m / max(k, 1))
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, upper, size=(W.shape[0], k)).astype(np.float64)

def print_matrix(M):
    rows = []
    for r in M:
        rows.append(",".join(f"{x:.4f}" for x in r))
    print("\n".join(rows))

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
        print("An Error Has Occurred")
        sys.exit(1)

def main():
    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        file_name = sys.argv[3]
        X = read_points(file_name)

        if goal == "sym":
            A = sm.sym(X)
            print_matrix(A)

        elif goal == "ddg":
            D = sm.ddg(X)
            print_matrix(D)

        elif goal == "norm":
            W = sm.norm(X)
            print_matrix(W)

        elif goal == "symnmf":
            W = sm.norm(X)
            H0 = init_H_from_W_mean(W, k, seed=1234)
            H  = sm.symnmf(H0, W, k, EPS, MAX_ITER, BETA)
            print_matrix(H)


        else:
            print("An Error Has Occurred")
            sys.exit(1)

    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()
