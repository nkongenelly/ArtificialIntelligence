import numpy as np
import sys

# determine Harshad - a number completely divisible by the sum of its digit
def func_seven(num):
    list_int = list(map(int, str(num).strip()))
    num_sum = sum(list_int)
    res = num % num_sum

    if res == 0:
        return f'{num} is a Harshad'
    else:
        return f'{num} is not a Harshad'

if __name__ == '__main__':
    num = np.random.randint(0,1000000,1)[0] #18
    res = func_seven(num)
    print(res)