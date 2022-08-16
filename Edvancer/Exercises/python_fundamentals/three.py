# takes a list of strings as input and removes strings until all remaining strings are of equal length
def func_three(input_list):
    sorted_list = list(sorted(input_list, key = len))

    same_length_res = []
    for id, str in enumerate(sorted_list):
        if id != len(sorted_list) -1 and len(str) == len(sorted_list[id+1]):
            same_length_res.extend([str, sorted_list[id+1]])
            break
    return same_length_res

if __name__ == '__main__':
    input_list = ['abc','abcd','acbde','acbgeh','qwert']
    res = func_three(input_list)
    print(res)
