import numpy as np
import math 

softmax_outputs= np.array([[0.7, 0.2, 0.1],
							[0.5, 0.1, 0.4],
							[0.02, 0.9, 0.08]])
target_output = [0, 1, 1]

# accuracy calculations
predictions = np.argmax(softmax_outputs, axis=1)
accuracy = np.mean(predictions == target_output)

print(accuracy)

# loss = -(math.log(softmax_output[0]) * target_output[0] + 
# 		math.log(softmax_output[1]) * target_output[1] + 
# 		math.log(softmax_output[2]) * target_output[2])

# print(loss)
# print(-math.log(0.7))


# layer_outputs = [[4.8, 1.21, 2.385],
# 				 [8.9, -1.81, 0.2],
# 				 [1.41, 1.051, 0.026]]
# # exponential capping overflow
# exp_values = np.exp(layer_outputs - np.max(layer_outputs, axis=1, keepdims=True))
# # print(exp_values)

# # normalization
# norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# print(norm_values)

