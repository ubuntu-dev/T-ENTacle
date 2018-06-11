# Goal: preprocess well names so they match (fuzzy match names)
from typing import List, Any

import pandas as pd
import numpy as np
import csv
from fuzzywuzzy import fuzz
import re
import collections
from pandas.io.json import json_normalize

# read csv files into a dataframe
df_extracted = pd.read_csv('stats_correct.csv')
df_gt = pd.read_csv('CompletionsDataExtract.csv')


# extract all unique rows in "Well Name" columns
wellname_extracted = list(set(df_extracted.loc[:, 'Well Name']))
wellname_gt = list(set(df_gt.loc[:, 'Well Name']))

#the extracted wellnames need some preprocessing. 
def clean_well_name(name):
    """
    The well names don't match the given names well, so some text cleaning needs to be performed. 
    :param name, String. Represents a well name
    :return name, String. The cleaned well name
    """
    name = re.sub("^cc_", "", name).replace("_", " ")
    patterns = [re.compile("(Well (History|Reports))$"), re.compile("Completion(s)? (Reports|Rpts)$"), re.compile("Completions$"), 
                re.compile("Rpts$"), re.compile("Well History Report(s)?$"), re.compile("REPORT$")]
    for p in patterns:
        name = re.sub(p, "", name).strip()
    return name

norm_to_ext_map= {clean_well_name(name): name for name in wellname_extracted}
wellname_extracted = list(norm_to_ext_map.keys())

# sort this list alphabetically (case insensitive) since lists are mutable
wellname_extracted = sorted(wellname_extracted, key=str.casefold)
wellname_gt = sorted(wellname_gt, key=str.casefold)

print("Getting well name mapping")
# map the extracted names to the names in the ground truth file
well_mapping = {}
#iterate through all the ground truth well names to check if they have a match
for i in range(len(wellname_gt)):
    gt_tokens = wellname_gt[i].split()

    candidates = [] #store all extracted wells that could be matches - determined by first word
    #print("Finding match for ", wellname_gt[i])
    #since the lists are sorted, advance to the correct letter of the alphabet
    for j in range(len(wellname_extracted)):        
        try:
            while wellname_extracted[j][0] < wellname_gt[i][0] and j < len(wellname_extracted):
                j += 1
        except IndexError:
            print(IndexError)
            print(j)
            print(len(wellname_extracted))
        
        #print("Checking ", wellname_extracted[i])
        #check if the first word matches
        ext_tokens = wellname_extracted[j].split()
        if ext_tokens[0].lower() == gt_tokens[0].lower():
            candidates.append(wellname_extracted[j])

        #stop looking for candidates once we hit the next letter of the alphabet
        if wellname_extracted[j][0].lower() > wellname_gt[i][0].lower():    
            if not candidates:
                #found nothing
                break
            elif len(candidates) == 1:
                #there is only one option
                well_mapping[norm_to_ext_map[candidates[0]]] = wellname_gt[i] #add it to the mapping

            else:
                #get the highest matching based on token set out of all the candidates
                scores = [(name, fuzz.token_set_ratio(wellname_gt[i].lower(), name.lower())) for name in candidates]
                sorted_scores = sorted(scores, key=lambda x: x[1], reverse = True)
                #print("all matches ", sorted_scores)
                best_match = sorted_scores[0][0]
                well_mapping[norm_to_ext_map[best_match]] = wellname_gt[i]
            
            break

print("Transforming the data")
# add a new column to the dataframe with the "corrected" column names
def map_wells(original):
    """
    Gets the normalized well name if it exists
    :param original, String. The name of the well we extracted
    :return String. The true name of the well or none if it doesn't exist.
    """
    if original in well_mapping.keys():
        return well_mapping[original]
    else:
        return None

df_extracted["Well Name Normalized"] = df_extracted["Well Name"].apply(lambda row: map_wells(row))

# for each well, get list of nums
values = {well_name: {} for well_name in well_mapping.values()}
#columns to iterate through
cols = list(df_extracted.columns)
cols.remove("Well Name")
cols.remove("Well Name Normalized")

for well in well_mapping.values():
    for col in cols:
        values[well][col] = []
        for item in df_extracted.loc[df_extracted["Well Name Normalized"] == well, col]:
            if not pd.isnull(item):
                if type(item) == str:
                    temp = item.split("\n")
                else:
                    temp = [item]
                for num in temp:
                    try: 
                        num = float(num)
                    except ValueError as e:
                        pass
                    if num not in values[well][col]:
                        values[well][col].append(num)


#calculate the precision and recall
recall = []
precision = []
i = 0
for well in well_mapping.values():
    recall.append({"Well Name" : well})
    precision.append({"Well Name" : well})
    for col in cols:
        if col == "Operator":
            recall[i]["Operator Name"] = values[well][col][0]
            precision[i]["Operator Name"] = values[well][col][0]
        predicted = set(values[well][col])
        true_col = set(df_gt[df_gt["Well Name"]==well][col].unique())
        tp = len(predicted.intersection(true_col))
        if true_col:
            recall[i][col] = tp/len(true_col)
        else:
            recall[i][col] = 0
        if predicted:
            precision[i][col] = tp/len(predicted)
        else:
            precision[i][col] = 0
    i += 1

recall_df = pd.DataFrame(recall)
precision_df = pd.DataFrame(precision)

recall_df.to_csv("recall_all_vendors_KE_original.csv")
precision_df.to_csv("precision_all_vendors_KE_original.csv")

#get an overall average
def get_avg(df):
    df_avg = df.drop(["Well Name", "Operator"], axis = 1).groupby("Operator Name").mean()
    df_avg["mean"] = df_avg.mean(axis = 1)
    return df_avg

recall_avg_df = get_avg(recall_df)
precision_avg_df = get_avg(precision_df)

recall_avg_df.to_csv("recall_avg_KE_original.csv")
precision_avg_df.to_csv("precision_avg_KE_original.csv")

        