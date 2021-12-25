# NEURAL NETWORKS IN 9 STEPS

# 1. INITIALIZATION
import numpy as np

np.random.seed(42) # for reproducibility

# 2. DATA GENERATION
X_num_row, X_num_col = [2, 10000] # Row is no. of feature, col is no. of datum points
X_raw = np.random.rand(X_num_row, X_num_col) * 100

print(X_raw[:5])