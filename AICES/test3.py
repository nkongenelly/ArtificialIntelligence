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
# print([X_train[row,:].mean() for row in range(X_num_row)])
# print([X_train[row,:].std() for row in range(X_num_row)])

# print([y_train[row,:].mean() for row in range(y_num_row)])
# print([y_train[row,:].std() for row in range(y_num_row)])

# 5. NEURAL NETWORK CONSTRUCTION
class layer:
	def __init__(self, layer_index, is_output, input_dim, output_dim, activation):
		self.layer_index = layer_index # zero indiccates input layer
		self.is_output = is_output # true indicates output layer
		self.innput_dim = input_dim
		self.output_dim = output_dim
		self.activation = activation

		# the multiplication constant is sorta arbitrary
		if layer_index !=0:

			self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2/input_dim)
			self.b = np.random.randn(output_dim, 1) * np.sqrt(2/input_dim)

layers_dim = [X_num_row, 4, 4, y_num_row] # input layer --- hidden layers --- out[ut layers
neural_net = []
# print(layers_dim)

# Construct the net layer by layer
for layer_index in range(len(layers_dim)):
	if layer_index == 0: # if input layer
		neural_net.append(layer(layer_index, False, 0, layers_dim[layer_index], 'irrelevant'))
	elif layer_index+1 == len(layers_dim): #if output layer
		neural_net.append(layer(layer_index, True, layers_dim[layer_index-1], layers_dim[layer_index], activation='linear'))
	else:
		neural_net.append(layer(layer_index, False, layers_dim[layer_index-1], layers_dim[layer_index], activation='relu'))
# print(neural_net)

# Simple check on overfitting
pred_n_param = sum([(layers_dim[layer_index]+1) * layers_dim[layer_index+1] for layer_index in range(len(layers_dim)-1)])
act_n_param = sum([neural_net[layer_index].W.size + neural_net[layer_index].b.size for layer_index in range(1,len(layers_dim))])
print(f'Predicted number of hyperparameters: {pred_n_param}')
print(f'Actual number of hyperparameters: {act_n_param}')
print(f'Number of data: {X_num_col}')

if act_n_param >= X_num_col:
	print('it will overfit')
	raise Exception('it will overfit.')