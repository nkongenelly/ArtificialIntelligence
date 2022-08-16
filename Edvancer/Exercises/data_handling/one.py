import pandas as pd
import numpy as np

class FuncOne:
    def func_one(self, d):
        d['max'] = np.absolute(d["x"] - d["y"])
        max = d['max'].max()

        print(d)
        print(d[d['max'] == max])
        index = d.index[d['max'] == max].tolist()[0]
        id = d._get_value(index, 'id')
        x = d._get_value(index, 'x')
        print('id corresponding to maximum absolute difference between x and y : ' + str(id))
        print('number of rows with value of x strictly higher than ' + str(x) +' = ' + str(len(d[d['x'] > x])))
        return id

obj = FuncOne()
d = pd.DataFrame({'id': np.random.choice(range(1, 100), 30, replace= False), 'x': np.random.randint(1,100,30), 'y': np.random.randint(1, 100, 30)})
res = obj.func_one(d)

