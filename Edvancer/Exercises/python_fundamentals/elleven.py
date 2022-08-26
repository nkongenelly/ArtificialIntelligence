
def func_elleven(sample):    
    sample_unique = set(sample)
    ans ={}
    for i in sample_unique:
        ans.update({i:sample.count(i)})
    return ans
if __name__ == '__main__':
    sample = ['a','b','a','a','b','c','c','a','b'] 
    res = func_elleven(sample)
    print(res)