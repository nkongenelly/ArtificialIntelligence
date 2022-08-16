import numpy as np

def func_five(random_dict):
    values = list(random_dict.values())
    x = np.array(values)
    unique = np.unique(x)
    return len(unique)

if __name__ == '__main__':
    random_dict = {f't{i}':np.random.randint(1,101) for i in range(1, 101)}
    res = func_five(random_dict)
    print(res)