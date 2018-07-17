# experiments for record detection using pattern analysis

import numpy as np
from collections import deque
import pandas as pd
import ast

def mode_tup(arr):
    arr=[tuple(a) for a in arr]
    mode=max(set(arr), key=arr.count)
    return list(mode)

def get_patt(ent, arr):
    n=0
    count = 0
    pattern = []
    flag=False
    while n <= len(arr):
        if n==len(arr):
            pattern.append(count)
        elif arr[n]==ent:
            pattern.append(count)
            count=0
            pattern.append(str(arr[n]))
            flag=True
        else:
            count+=1

        n += 1
    if flag:
        return pattern
    else:
        return []

def align_separate(arr, gt):
    rec_ind=[]
    last_found=-1
    i=0
    for j in range(len(arr)):
        if arr[j] not in gt:
            continue
        notfound = True
        while notfound:
            if i==len(gt):
                i=0
                rec_ind.append(last_found)
            if arr[j]==gt[i]:
                last_found=j
                notfound=False
            i+=1

    return rec_ind

def rec_separate(data,stage):

    # data=np.array([1,2,3,2,5,2,3,2,4,5,1,2,3,2,4,5,1,2,3,2,4,5,1,2,3,2,4,5])
    data=np.array(data)
    # stage=3

    entities=list(set(data))
    splits=np.split(data,np.argwhere(data==stage).flatten())


    per_entity_pattern={}
    for i in range(1, len(splits)-1):
        split=splits[i][1:]
        # print(split)
        for entity in entities:
            patt=list(np.argwhere(split==entity).flatten())
            if len(patt)!=0:
                if entity not in per_entity_pattern.keys():
                    per_entity_pattern[entity]=[]

                per_entity_pattern[entity].append(patt)

    # print(per_entity_pattern)

    # create the best blueprint

    blueprint={}
    for ent, value in per_entity_pattern.items():
        # print(ent, value)
        mode=mode_tup(value)
        for mod in mode:
            blueprint[mod]=ent

    indexes=list(sorted(blueprint.keys()))
    bp=[stage]
    for i in indexes:
        bp.append(blueprint[i])
    # print(bp)

    # decide the starting point, then rotate the blue print accordingly
    start_index=len(splits[-1])
    bp=deque(bp)
    bp.rotate(len(bp)-start_index)
    bp=list(bp)
    # print(bp)

    # sync a new pattern according to the blue print

    # test=[1,2,3,2,5,2,3,2,4,5,1,2,3,2,5,1,2,3,2,4,1,2,3,2,4,5]
    records=np.split(data, np.argwhere(data==bp[0]).flatten())
    # print(records)
    global_index=0
    new_records=[]
    record_indices=[]

    for record in records:
        if len(record)>len(bp):
            # print('potential multiple records')
            # print(record)
            rec_inds=align_separate(record, bp)
            begin=0
            for rec_ind in rec_inds:
                # print(rec_inds)
                rec=list(record[begin:rec_ind])
                new_records.append(rec)
                begin=rec_ind
                record_indices.append(list(range(global_index, global_index+len(rec))))
                global_index=global_index+len(rec)
        elif len(record)!=0:
            rec=list(record)
            new_records.append(rec)
            record_indices.append(list(range(global_index, global_index + len(rec))))
            global_index=global_index + len(rec)

    return new_records,record_indices

# create the 'best' blueprint or the patterns with all entities together
##  now filter this blueprint to include only the ones that have a one-to-one perfect interleaving with 'stage'
# now using multiple pattern instances (from multiple documents) find the starting point in the blueprint
# now start from that decided entity and create records according to the blueprint

# def build_report_w_smart_align():
#
#     stage_name="stage"
#
#     all_patterns=pd.read_csv('all_patterns.csv',index_col=0)
#     all_patterns = all_patterns.replace(np.nan, '', regex=True)
#     learned_patterns=pd.read_csv('learned_patterns.csv',index_col=0)
#     learned_patterns = learned_patterns.replace(np.nan, '', regex=True)
#
#     # initialize by random entity names
#     all_patterns['entity_name'] = pd.Series(np.random.randn(len(all_patterns)), index=all_patterns.index)
#     # print(all_patterns['entity_name'])
#
#     all_pattern_ids=[]
#     print(learned_patterns)
#     for index,row in learned_patterns.iterrows():
#         entity_name=row['entity_name']
#         pattern_ids=ast.literal_eval(row['pattern_ids'])
#         all_pattern_ids.extend(pattern_ids)
#         for id in pattern_ids:
#             all_patterns.loc[all_patterns['pattern_id']==id,'entity_name']=entity_name
#
#     all_patterns=all_patterns[all_patterns['pattern_id'].isin(all_pattern_ids)]
#     all_patterns=all_patterns.reset_index(drop=True)
#
#
#     entities=set(all_patterns['entity_name'])
#     series=pd.DataFrame()
#
#     for entity in entities:
#         instances_orders=ast.literal_eval(str(list(all_patterns[all_patterns['entity_name']== entity]['instances_orders'])[0]))
#         instances = ast.literal_eval(str(list(all_patterns[all_patterns['entity_name'] == entity]['instances'])[0]))
#         entity_names=[entity] * len(instances)
#         df=pd.DataFrame(data={"instances_orders":instances_orders,"instances":instances,"entity_name":entity_names})
#         series=pd.concat([series,df])
#
#     series=series.sort_values(by=['instances_orders'])
#     series=series.reset_index(drop=True)
#     print(list(series['entity_name']))
#     records,indices=rec_separate(list(series['entity_name']),stage_name)
#     print(records,indices)
#
#     separated_records=pd.DataFrame(columns=list(entities))
#
#     for record_index in indices:
#         tempdict={}
#         for index in record_index:
#             entity_name=series.loc[index]['entity_name']
#             instances = series.loc[index]['instances']
#             tempdict[entity_name]=instances
#         separated_records=separated_records.append(tempdict,ignore_index=True)
#
#     separated_records.to_csv('aligned_records.csv')


