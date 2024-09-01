import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from functions_PIMD import *
from matplotlib.pyplot import figure
from matplotlib.legend import Legend

###### Learning Pandas Commands ######
# colvar=pd.read_csv("log.lammpsc.0", sep='[\s,]{2,20}')
# df = pd.read_csv("log.lammpsc.0", delimiter='\s+')
# print(df.head(5))
#                       Read header
# print(df.columns)
#                        Read each Column
# print(df['PotEng'])
#                         Read each Row
# print (df.iloc[0:2])
#                        Read specific locations (Row, Column)
# print (df.iloc[1,4])         '                   # Second Row 5th Column (PotEng)
# print (df.loc[df['PotEng'] > 1])               # All the PotEng that is greater than 1
# print (df.describe())                              # Shows the mean, stdv, etc off all the data
# print (df['PotEng'] + df['PotEng'])                # Adding the Column of the Pot Energy
# df.to_csv('new_file.csv', index=False)            # Create a csv with modified data. the Index is to delelte the indexes it adds add the left.
#to_excel
#df.to_csv('new_file.txt', index=False, sep='\t')

#How to use the format command
# file = []
# file = ['log.lammps.{}'.format(k) for k in range(0, 5)]
# print(file)
#################################################################################Above Learning PANDAS










