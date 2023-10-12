from itertools import product
import pandas as pd

location = list(range(0, 10000, 150))
temp = location[1:]
for i in range(0, len(temp)):
    temp[i] = -(temp[i])
location = location + temp
locaXY = list(product(location, location))

rows = []
for i in range(0, len(locaXY)):
    rows.append({'id': 'm' + str(i), 'x': locaXY[i][0], 'y': locaXY[i][1]})

mec = pd.DataFrame(rows, columns=['id', 'x', 'y'])
mec.to_json('MEC_server.json', orient='table')
