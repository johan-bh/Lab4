import numpy as np
from scipy import stats
import pandas as pd

f = open("results.txt","r")
lines = f.readlines()

data = []
for line in lines:
    # Strip lines
    stripped_line = line.strip().rstrip(',').strip('[]')
    # Create a list split by comma
    linelist = stripped_line.split(", ")
    # Convert all elements in list to float
    line_data = [ float(n) for n in linelist ]
    data.append(line_data)


df = pd.DataFrame({'50': data[0], '100': data[1], '150': data[2], '200': data[3]})
