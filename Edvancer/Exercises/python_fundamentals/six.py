def func_six(radius):
    area_of_big_square = (2*radius)**2
    area_of_small_square = 2*radius**2

    return area_of_big_square - area_of_small_square

if __name__ == '__main__':
    res = func_six(7)
    print(res)
