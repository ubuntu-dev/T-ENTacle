# Goal: Get all the items in csv file s.t. we skip over empty or NaN cells

import pandas as pd  #imports the pandas library and aliasing as pd
import numpy as np
import csv  #converting new_list to csv file

df = pd.read_csv('stats_correct.csv')  #read the csv file into  a dataframe 
# print(df.head())  #prints first five rows
# print(df.isnull())  #returns true for NaN items/cells

new_list = []

for col in df.columns:
    # boolean: filters out NaN representing empty cells
    col_new = df[df[col].notnull()][col]

    #print(col)  #prints all the column headers
    for item in col_new:  #points to an item/cell under column header in col_new
        if isinstance(item, str):  #checks if item is a string
            temp = item.split("\n")  #split those strings separated by \n
        else:
            temp = [item]  #else get that item anyway

        print(item) #prints all items in csv file excluding column headers

        # for num in temp:
        #     print(num)
        #     num = float(num)
        #     # break
        #     if num not in new_list:
        #         new_list.append(num)

