import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

gt=pd.read_csv('final.csv')
gt=gt[gt['Operator'].isin(['ANADARKO E&P ONSHORE LLC','ANADARKO E&P'])]
predict=pd.read_csv('report_smart_align.csv')

# well_name={
#     "Banshee Well Reports.pdf":"Banshee",
#     "BANSHEE 56-3-28 UNIT 1H":"Banshee",
#     "BEANS STATION 53-4-5 1H Completion Rpts.pdf":"Bean Station",
#     "BEANS STATION 53-4-5 1H":"Bean Station",
#     "BULL RUN STATE 55-4-22 UNIT 1H":"Bull Run",
#     "AVALANCHE 29-40 UNIT 1H Completion Reports.PDF":"Avalanche",
#     "AVALANCHE 29-40 UNIT 1H":"Avalanche",
#     "cc_MAGIC STATE 56-3-39 UNIT 1H Completion Rpts.pdf":"Magic State",
#     "MAGIC STATE 56-3-39 UNIT 1H":"Magic State",
#     "SEAHAWK 57-2-1 1H":"Seahawk",
#     "cc_SHILOH STATE 55-4-34 UNIT 1H Completion Rpts.pdf":"Shiloh State",
#     "SHILOH STATE 55-4-34 UNIT 1H":"Shiloh State",
#     "cc_SILVERTIP 76-10 UNIT H 1H Completion Rpts.pdf":"Silvertip",
#     "SILVERTIP 76-10 UNIT H 1H":"SilverTip",
#     "cc_WOLFHOUND 56-3-25 1H Completion Rpts.pdf":"Wolfhound",
#     "WOLFHOUND 56-3-25 1H":"Wolfhound",
#     "cc_ZION 55-1-17 UNIT 1H Completion Rpts.pdf":"Zion",
#     "ZION 55-1-17 UNIT 1H":"Zion",
#     "EBONY 55-1-40 UNIT 1H Completions.pdf":"Ebony",
#     "GROWLER 56-3-23 1H Completion Rpts.pdf":"Growler",
#     "IVY MOUNTAIN 53-4-3 UNIT 1H Completion Rpts.pdf":"Ivy Mountain",
#     "IVY MOUNTAIN 53-4-3 UNIT 1H":"Ivy Mountain",
#     "JACKALOPE 56-3-11 UNIT 1H Completion Rpts.pdf":"Jacklope",
#     "JACKALOPE 56-3-11 UNIT 1H":"Jacklope",
#     "KAYCEE 57-3-41 1H Completion Rpts.pdf":"Kaycee",
#     "KAYCEE 57-3-41 1H":"Kaycee",
#     "Manticore 1H Well Reports.pdf":"Manticore",
#     "MANTICORE STATE 55-3-3 UNIT 1H":"Manticore",
#     "MEDICINE BOW 56-1-23 1H Completion Rpts.pdf":"Medicine Bow",
#     "MEDICINE BOW 56-1-23 1H":"Medicine Bow",
#     "NESSIE 56-2-35 UNIT 1H Completion Rpts.pdf":"Nessie",
#     "NESSIE 56-2-35 UNIT 1H":"Nessie",
#     "SILVERTIP 76-14 UNIT U 1H Completions Rpts.pdf":"Silvertip_2",
#     "SILVERTIP 76-14 UNIT U 1H":"Silvertip_2",
#     "PELICAN BAY 34-180 UNIT 1H":"Pelican Bay",
#     "WOOD LAKE 54-4-23 1H Completion Rpts.pdf":"Wood Lake",
#     "CHIMERA STATE 56-3-7 UNIT 1H":"Chimera"
# }

# ====mappings to standardize the Well name. The well name is synonymous to document name in our repost,
# whereas the ground truth report contains no document name but well name. So we convert both to a common name and put it in a new column called 'Well'
# Note: Select from above and add accordingly to below.
well_name={
    "BEANS STATION 53-4-5 1H Completion Rpts.pdf":"Bean Station", # for our report
    "BEANS STATION 53-4-5 1H":"Bean Station" # for gt report
}

def temp(v):
    if v in well_name.keys():
        return well_name[v]
    else:
        return v

gt['Well'] = gt['Well Name'].map(temp)
gt = gt[gt['Well'].isin(well_name.values())] # also filters out the Wells that are not in the well_name mapping, can use this to remove Wells for which we did not run our software
predict['Well'] = predict['document'].map(temp)


# ====match the names of mismatched columns
matchings={
    'Total vol slick water gal':'Total  vol slickwater, gal'}

predict = predict.rename(columns=matchings)
print(predict.columns)

common=list(set(gt.columns).intersection(set(predict)))
# print(gt.columns)
# print(predict.columns)
# print('common columns between the two reports'.common)
# print('Alert! columns that are in our report but are not in the common columns',set(predict)-set(common))

# ====remove the commas from the numbers, as the gt report does not have the commas in the numbers
def temp2(v):
    return str(v).replace(',','')

for column in predict.columns:
    if column in ['document', 'stage number']:
        continue
    predict[column] = predict[column].map(temp2)


# ====only keep the columns/entities that we learned
gt=gt[common]
predict=predict[common]

# ====save for eyballing
gt.to_csv('gt.csv')
predict.to_csv('predict.csv')


# ====align the two report using "Well" and "stage number" columns.
# The merge will add a suffix to each entity depending on if it belonged to gt or predict report.
res=gt.merge(predict, on=['Well', 'stage number'], how='outer', suffixes=['_gt', '_predict'])
res.to_csv('compare.csv')
res=res.fillna(0)

# ====for each entity find the recall and precision
def match(v1,v2):
    # find how many items (corresponding) a same in two lists
    count=0
    for a,b in zip(v1,v2):
        if a==b:
           count+=1
    return count

rearrage_cols=['stage number', 'Well'] # this list makes sure the columns are in the order that we want it to.

for columns_name in common:
    if columns_name in ['stage number','Well']:
        continue
    print(columns_name)
    rearrage_cols.append(columns_name+'_gt')
    rearrage_cols.append(columns_name + '_predict')

    gt_arr=list(res[columns_name+'_gt'])
    gt_arr=[float(a) for a in gt_arr]

    predict_arr=list(res[columns_name + '_predict'])
    predict_arr = [float(a) for a in predict_arr]

    # prs=precision_recall_fscore_support(gt_arr, predict_arr, average='macro')
    # print(prs)

    print(match(gt_arr, predict_arr),' match, out of ',len(gt_arr))

res=res[rearrage_cols]
res.to_csv('compare.csv')
