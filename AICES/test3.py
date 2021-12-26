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

# 3.TRAINING-TEST SPLITTING
train_ratio = 0.7
num_train_datum = int(train_ratio * X_num_col)
X_raw_train = X_raw[:, 0:num_train_datum]
X_raw_test = X_raw[:, num_train_datum:]

y_raw_train = y_raw[:, 0:num_train_datum]
y_raw_test = y_raw[:, num_train_datum:]

# 4. DATA STANDADIZATION
# Data standardization is the process of rescaling the attributes so that they have mean as 0 and variance as 1
# The ultimate goal to perform standardization is to bring down all the features to a common scale without distorting the differences in the range of the values. 

class scaler:
	def __init__(self, mean,std):
		self.mean = mean
		self.std = std

def get_scaler(row):
	mean = np.mean(row)
	std = np.std(row)
	return scaler(mean, std)

def standardize(data, scaler):
	return (data - scaler.mean) / scaler.std

def unstanderdize(data, scaler):
	return (data * scaler.std) + scaler.mean

# Construct scalers (column scalers) from training set

X_scalers = [get_scaler(X_raw_train[row, :]) for row in range(X_num_row)]
X_train = np.array([standardize(X_raw_train[row,:], X_scalers[row]) for row in range(X_num_row)])

y_scalers = [get_scaler(y_raw_train[row, :]) for row in range(y_num_row)]
y_train = np.array([standardize(y_raw_train[row,:], y_scalers[row]) for row in range(y_num_row)])

# Apply those scalers to testing set
X_test = np.array([standardize(X_raw_test[row,:], X_scalers[row]) for row in range(X_num_row)])
y_test = np.array([standardize(y_raw_test[row,:], y_scalers[row]) for row in range(y_num_row)])

# check if data has been standardized
print([X_train[row,:].mean() for row in range(X_num_row)])
print([X_train[row,:].std() for row in range(X_num_row)])

print([y_train[row,:].mean() for row in range(y_num_row)])
print([y_train[row,:].std() for row in range(y_num_row)])