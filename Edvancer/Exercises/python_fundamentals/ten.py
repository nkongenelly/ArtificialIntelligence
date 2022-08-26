def func_ten(addressess):
    cities = []
    for addre in addressess:
        new_addr = (addre.split(','))[-1]
        cities.append(new_addr)
    return cities

if __name__ == '__main__':
    addressess = ['H-73, MDT, Powai , Mumbai' , '1604, SS, Hyderabad' ,'B block 73, Adyar, Chennai']
    res = func_ten(addressess)
    print(res)