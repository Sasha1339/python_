import numpy as np
import matplotlib.pyplot as plt
import os
import re

data_dir = "../data"

fld = open("LD2011_2014.txt", "r")
data = []
cid = 250
count = 0
for line in fld:
    count += 1
    if count == 1 :
        continue
    cols = []
    for x in line.strip().split(";")[1:]:
        cols.append(float(re.sub(",", ".", x)))
    data.append(cols[cid])
fld.close()
print(data)

NUM = 1000
plt.plot(range(NUM), data[0:NUM])
plt.show()
np.save("LD_250.npy", np.array(data))
