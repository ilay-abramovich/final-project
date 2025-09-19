#!/usr/bin/env python3
# analysis.py — compare silhouette of SymNMF vs KMeans

import sys
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from symnmf import *
from kmeans import run_kmeans
import numpy as np

def init_H_from_W_mean(W: np.ndarray, k: int, seed: int = 1234) -> np.ndarray:
    m = float(W.mean())
    upper = 2.0 * np.sqrt(m / max(k, 1))
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, upper, size=(W.shape[0], k)).astype(np.float64)

def read_points(path):
    X = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            X.append([float(v) for v in s.split(",")])
    return np.asarray(X, dtype=np.float64)

def hard_labels(H):
    return np.argmax(H, axis=1)

def main():
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        file_name = sys.argv[2]
        X = read_points(file_name)

        # SymNMF
        H = run_symnmf(X, k)
        lbl_nmf = hard_labels(H)

        # KMeans
        X, lbl_km = run_kmeans(X, k)

        s_nmf = silhouette_score(X, lbl_nmf)
        s_km = silhouette_score(X, lbl_km)

        print(f"nmf: {s_nmf:.4f}")
        print(f"kmeans: {s_km:.4f}")

    except Exception as e:
        print(e)
        print("An Error Has Occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
