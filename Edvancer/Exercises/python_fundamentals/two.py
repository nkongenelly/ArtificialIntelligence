# function takes any input
# replace letters that are keys with their values
def func_two(str):
    replacement = {'a':'e','b':'$','c':'q','d':'p','j':'n'}
    replaced = [replacement[i] if i in replacement else i for i in str ]
    return ''.join(replaced)

if __name__ == '__main__':
    str = 'abjected'
    res = func_two(str)
    print(res)