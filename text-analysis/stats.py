import knowledge_extractor
import os
from collections import Counter
import json
from fuzzywuzzy import fuzz
import datetime
import pandas as pd
import numpy as np
import argparse

def run_extractor(vendors, PATH):
    """
    Run knowledge_extractor on all vendors returning a JSON of the total entities found.
    """
    agg_df = pd.DataFrame()
    for vendor in vendors:
        keyDict = {}
        for n, file in enumerate(os.listdir(PATH + "/" + vendor + "/")):
            if '.DS_Store' in file:
                continue
            fullpath = os.path.join(PATH, vendor) + '/' + file
            print(fullpath)
            resultDict, df = knowledge_extractor.create_csv(fullpath)
            agg_df=pd.concat([agg_df,df],ignore_index=True)
            print("Number of Entities Found: %d" % (len(resultDict.keys())))

            keyDict = Counter(keyDict) + Counter(resultDict)
            print("Total Number of Entities Found: %d" % (len(keyDict.keys())))
            print(len(keyDict.keys()), sum(keyDict.values()))
        jsonDict = json.dumps(keyDict)
        file = open("%s_Dict.json" % (vendor), 'w')
        file.write(jsonDict)
        file.close()
        return agg_df

def run_GQ_labeler(vendors, PATH, out):
    """
    Run Grobid-Quantities on all vendors
    """
    for vendor in vendors:
        keyDict = {}
        for n, file in enumerate(os.listdir(PATH + vendor)):
            if '.DS_Store' in file:
                continue
            fullpath = PATH + vendor + '/' + file
            print(fullpath)
            resultDict = generate_data.run_entitylabeler(fullpath, out+file+".csv")
            print("Number of Entities Found: %d" % (len(resultDict.keys())))

            keyDict = Counter(keyDict) + Counter(resultDict)
            print("Total Number of Entities Found: %d" % (len(keyDict.keys())))
            print(len(keyDict.keys()), sum(keyDict.values()))
        jsonDict = json.dumps(keyDict)
        file = open("%s_Dict_GQ.json" % (vendor), 'w')
        file.write(jsonDict)
        file.close()

def get_score(data):
    """
    Return a score by fuzzy matching the entities to a dictionary of known entities.
    method 1: asitang's. method 2: annie's
    """
    grantsDict = {
        "Api": ["api", "api name", "api 10"],
        "Operator" : ["Operator"],
        "Well Name": ["Well Name"],
        "Completion": ["completion", "completion start date", "start date"],
        "Stage Top perf": ["Stage Top Perf", "Perf", "Perf'd", "Perf Intervals"],
        "Stage Bottom perf": ["Stage Bottom Perf", "Perf", "Perf'd", "Perf Intervals"],
        "Gun length": ["gun", "gun length", "cluster length", "cluster len", "gun len"],
        "shots per ft": ["shots per ft", "spf"],
        "number of perfs": ["number of perfs", "number of perfs per stage"],
        "Total vol slickwater": ["Total vol slickwater", "slickwater", "SW"],
        # Empty column: "Total vol linear gel": ["Total vol linear gel", "linear gel", "gel loading"],
        # Empty column: "Total vol x-link": ["Total vol x-link"],
        "total fluid": ["total fluid", "total clean fluid"],
        # Empty column: "Fluid Vol QC"
        # Empty column: "#gel loading"
        "stage number": ["stage #", "stg", "stg #"],
        "max prop conc": ["max prop conc", "prop con", "ppa", "ppg", "max ppg"],
        # Empty column: "Total 200mesh": ["total 200mesh", "200mesh"],
        "Total 100mesh": ["total 100mesh", "100mesh"],
        # Empty: "Total 40_70 mesh": ["total 40_70 mesh", "40_70 mesh"],
        # Empty: "Total 30_50 mesh": ["total 30_50 mesh", "30_50 mesh"],
        # Empty: "Total 20_40 mesh": ["total 20_40 mesh", "20_40 mesh"],
        # Empty: "Total 16_30 mesh": ["total 16_30 mesh", "16_30 mesh"],
        "Total proppant": ["total proppant"],
        "Total prop QC": ["Total proppant QC", "total prop"],
        # Empty: "Total white sand": ["total white sand", "total white sand lbs", "white sand", "100 mesh white"],
        # Empty: "Total brown sand": ["total brown sand", "total brown sand lbs", "brown sand"],
        # Empty: "Total resin coated": ["total resin coated", "resin coated", "resin"],
        # Empty: "Total ceramic sand": ["ceramic sand", "total ceramic sand"],
        "pump rate avg": ["pump rate avg", "pump rate average, average treating rate", "average pump rate", "atr", "apr", "ar", "avg. rate"],
        "pump rate max": ["pump rate max", "max treating rate", "max pump rate", "mtr", "mpr", "mr", "max rate"],
        "avg treating pressure": ["average treating pressure", "atp", "ap", "avg press"],
        "max treating pressure": ["max treating pressure", "mtp", "mp", "max pressure", "max press"],
        "ISIP": ["ISIP", "ISIP FINAL"],
    }
    extractedKeys = list(data.keys())
    print(extractedKeys)
    correctSize = len(grantsDict)
    found = False
    totalScore = 0
    foundList = []
    notFoundList = []
    dictFound = {}

    for correct in grantsDict.keys():
        for n, alias in enumerate(grantsDict[correct]):
            for extractedN in extractedKeys:
                sim_score = fuzz.ratio(alias.upper(), extractedN.upper())
                if sim_score >=80:
                    print("Correct Name: %s, Alias Found: %s, Original name: %s" % (correct.upper(), alias, extractedN))
                    dictFound[extractedN] = (correct.upper(), extractedN)
                    found = True
        if found:
            totalScore+=1
            found = False
            foundList.append(correct.upper())
        else:
            notFoundList.append(correct.upper())
    print(foundList)
    print(notFoundList)
    finalScore = ((float(totalScore)/float(correctSize))*100.0)
    print(("The accuracy was %f percent.") % (float(finalScore)))
    return dictFound

def match_entities(vendor):
    mapping={}
    if vendor=='Anadarko':
        mapping={'STAGE':['stage number'],
                'SLICKWATER (BBLS)':['Total  vol slickwater, gal'],
                 'TOTAL FLUID (BBLS)':['total fluid (gal)'],
                 'MAX PPG':['max prop conc (ppa)'],
                 '100 M':['Total 100mesh (lbs)','Total prop QC'],
                 'TOTAL PROP LBM':['total proppant (lbs)'],
                'ATR':['pump rate avg (bpm)'],
                 'MTR':['pump rate max (bpm)'],
                 'ATP':['avg treating pressure (psi)'],
                 'MTP':['max treating pressure (psi)'],
                 'ISIP (FINAL)':['ISIP (psi)'],
                 'ISIP PSIG':['ISIP (psi)']
                 }

    if vendor=='Cimarex':
        mapping={
            'Avg Rate (bpm)':['pump rate avg (bpm)'],
            'Max Rate (bpm)':['pump rate max (bpm)'],
            'Max rate (bpm)': ['pump rate max (bpm)'],
            'ATP':['avg treating pressure(psi)'],
            'Max(Psi, psi)':['max treating pressure (psi)'],
            'pressure(psi)':['max treating pressure (psi)'],
            'ISIP':['ISIP(psi)']
        }
    if vendor=='EOG':
        mapping={
        'Stage':['stage number'],
            'Proppant Total Fluid Open':['total fluid (gal)'],
            'Proppant EOG100 Mesh':['Total 100mesh (lbs)','Total prop QC'],
            'Total Proppant Max Prop':['total proppant (lbs)'],
            'Avg Max Pump Rate':['pump rate avg (bpm)'],
            'Max Rate':['pump rate maxm (bpm)'],
            'Max (PSI)':['avg treating pressure (psi)'],
            'Max (Psi)':['max treating pressure (psi)']
        }

    return mapping




def run_comparison(df,vendor=''):
    if vendor=='':
        dictFound=get_score(df)
    else:
        dictFound = match_entities(vendor)
        dictFound={k:' & '.join(dictFound[k]) for k in dictFound.keys()}
    dictFound['vendor']='Operator'
    dictFound['well'] = 'Well Name'
    df = df.filter(items=dictFound.keys())
    df = df.rename(index=str, columns=dictFound)
    df=df.groupby(df.columns, axis=1).agg(np.max) # aggregate the columns with same name
    return df


if __name__=="__main__":
    # Apply knowledge_extractor
    # TODO: Change to args
    # VENDORS = something called with args
    desc = """Calculates the number of entities found from Grant's list"""
    parser = argparse.ArgumentParser(description= desc)
    parser.add_argument('path_to_pdfs', metavar= "pdfs", type = str,
                        help= "Path to the directory containing one folder per vendor of pdfs")
    parser.add_argument("vendors", metavar = "v", nargs = "+", default=[],
                        help = "A list containing the names of the vendor folders")
    #PATH = "/Users/asitangm/Desktop/pdfs_de_chevey/test/"
    args = parser.parse_args()

    PATH = args.path_to_pdfs
    VENDORS = args.vendors

    all_df_matched = []
    all_df = []
    for vend in VENDORS:
        df_vend = run_extractor([vend], PATH)
        all_df.append(df_vend)
        df_matched = run_comparison(df_vend, vend)
        all_df_matched.append(df_matched)
    
    
    final_df_matched=pd.concat(all_df_matched,ignore_index=True)
    final_df_matched.to_csv('stats_correct.csv')
    final_df = pd.concat(all_df, ignore_index=True)
    final_df.to_csv('stats_all.csv')

    #Compare stats to vendors
    for name in VENDORS:
        print("********")
        print(name)
        print("********")
        json_data = open("%s_Dict.json" % (name)).read()
        data = json.loads(json_data)
        #get_score(name)
