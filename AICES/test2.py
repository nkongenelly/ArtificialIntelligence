import math

layer_outputs = [4.8, 1.21, 2.385]

E = math.e

exp_values = []

# exponential
for outputs in layer_outputs:
	exp_values.append(E**outputs)

# normalization
norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
	norm_values.append(value / norm_base);

print(norm_values)

