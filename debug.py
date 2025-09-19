import numpy as np
import symnmfmodule as sm  # your compiled extension

W = np.eye(3, dtype=np.float64)
H0 = np.random.rand(3, 2).astype(np.float64)
k = 2
eps = 1e-4
max_iter = 100

# Test 5-arg call (uses default beta)
H = sm.symnmf(H0, W, k, eps, max_iter)

# Test 6-arg call
beta = 0.3
H2 = sm.symnmf(H0, W, k, eps, max_iter, beta)
