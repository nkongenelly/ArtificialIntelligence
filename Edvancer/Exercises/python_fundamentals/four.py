def func_four(x):
    x = int(x) 
    b=0
    a=1
    c=x
    while c > 5 and c%5 !=0:
        b=(c-3*a)/5
        a+=1
        c-=3
    if c%5==0:
        res = 'Yes'
    else:
        res = 'No'
    return f'a={a} and {c}/5 results to {res}'

if __name__ == '__main__':
    res = func_four('7')
    print(res)