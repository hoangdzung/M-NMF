import numpy as np 
import pandas as pd 
import sys

npyfile = sys.argv[1]
csvfile = sys.argv[2]

edge_index = np.load(npyfile)
pd.DataFrame(np_array).to_csv(csvfile,index=None, header=['node_1','node_2'])
