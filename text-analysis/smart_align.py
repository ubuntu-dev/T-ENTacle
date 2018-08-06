import numpy as np
from collections import deque

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


def rec_separate(data, stage):

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
