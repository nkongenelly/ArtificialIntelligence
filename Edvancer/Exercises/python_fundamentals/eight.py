import numpy as np
# fibonacci series and give sum of een numbers within the limit given
def func_eight(num):
    n1 = 1
    n2 = 1
    fib = [1,1]
    count = 2
    while count < num:
        nth = n1 + n2
        fib.append(nth)
        n1 = n2
        n2 =  nth
        count += 1

    even_list = [i for i in fib if i%2 ==0]

    list_sum = sum(even_list)
    return {'sum':list_sum,'fibb':fib}

if __name__ == '__main__':
    num = np.random.randint(0,15,1)[0] #18
    res = func_eight(num)
    print(f'sum of fibonacci series {res["fibb"]} is {res["sum"]}')
