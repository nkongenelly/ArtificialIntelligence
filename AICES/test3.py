# NEURAL NETWORKS IN 9 STEPS

# 1. INITIALIZATION
import numpy as np

np.random.seed(42) # for reproducibility

# 2. DATA GENERATION
X_num_row, X_num_col = [2, 10000] # Row is no. of feature, col is no. of datum points
# Rand generates values between 0 andd 1, we multiply by 100 to make the max 100
X_raw = np.random.rand(X_num_row, X_num_col) * 100 
# print(X_raw[0,:].shape)


# for input a and b, output is a+b, a-b and |a-b|
y_raw = np.concatenate(([(X_raw[0,:] + X_raw[1,:])],
						[(X_raw[0,:] - X_raw[1,:])],
						np.abs([(X_raw[0,:] - X_raw[1,:])]))) 
y_num_row, y_num_col = y_raw.shape
# print(y_raw.shape)