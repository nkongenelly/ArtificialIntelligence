import random
import numpy as np

def function_one(initial_list):
    # remove all divisible by 3
    not_divisible_by_three = [i for i in initial_list if i%3 != 0 ]

    # subbtract 1 from all remaining values
    list_minus_one = [i-1 for i in not_divisible_by_three]

    # remove all divisible by 5 from the list
    not_divisible_by_five = [i for i in list_minus_one if i%5 != 0 ]

    return not_divisible_by_five

if __name__ == '__main__':
    # Generate a list of 50 random integers between 10 and 100
    initial_list = np.random.randint(10,101,50)
    res = function_one(initial_list)
    print(res)
    # print(len(res))
    # print(type(res))