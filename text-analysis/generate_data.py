#from spacy_entitylabeler import EntityLabeler
from entitylabeler import EntityLabeler
import requests
import tika
tika.initVM()
import PyPDF2
import os
from tika import parser
import pandas as pd
import shutil
from collections import Counter

unit_entities = {"ft" : ["Stage Top perf", "Stage Bottom perf", "cluster spacing", "stage length", "gun length"],
    "gal": ["Total  vol slickwater", "Total  vol linear gel",  "Total  vol  x-link", "total fluid", "Fluid Vol QC"],
    "lbsper1000gal" :[ "gel loading"],
    "ppa": ["max prop conc"],
    "lbs": ["Total 200mesh", "Total 100mesh",	"Total 40_70 mesh",	"Total 30_50 mesh",	"Total 20_40 mesh",
            "Total 16_30 mesh" , "total proppant",	"Total prop QC", "Total white sand", "Total brown sand",
             "Total resin coated", "Total ceramic sand"],
    "bpm": ["pump rate avg", "average treating rate","max treating rate", "max pump rate", "pump rate max"],
    "psi": ["avg treating pressure", "max treating pressure", "ISIP"]
                 }

url = "http://localhost:8060/processQuantityText"

#TODO ex1, ex2, ex3 moved to file. Add something to read from file


def split_pdf_pages(input_pdf_path, target_dir, fname_fmt=u"{num_page:04d}.pdf"):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(input_pdf_path, "rb") as input_stream:
        input_pdf = PyPDF2.PdfFileReader(input_stream)

        if input_pdf.flattenedPages is None:
            # flatten the file using getNumPages()
            input_pdf.getNumPages()  # or call input_pdf._flatten()

        for num_page, page in enumerate(input_pdf.flattenedPages):
            output = PyPDF2.PdfFileWriter()
            output.addPage(page)

            file_name = os.path.join(target_dir, fname_fmt.format(num_page=num_page))
            with open(file_name, "wb") as output_stream:
                output.write(output_stream)

### insert a path
# test_file = ""
# os.mkdir('./temp')
# split_pdf_pages(test_file, 'temp')

# for a small text snippet
# payload = {"text": ex3}
#
# r = requests.post(url, data=payload)
# print(r.status_code)
# if r.status_code == requests.codes.ok:
#     annotations = r.json()["measurements"]
#     #print(type(annotations))
#     test = EntityLabeler(ex3, annotations, unit_entities)
#
#     df = test.parse()
#     if df is not None and not df.empty:
#         with open("output.csv", "a") as f:
#             df.to_csv(f, index=False)

def run_entitylabeler(inpath, outpath):
    if not os.path.exists("./temp"):
        os.mkdir('./temp')
    split_pdf_pages(inpath, 'temp')
    print("Processing file ", inpath)
    with open(outpath, "w") as f:
        df = pd.DataFrame()
        pdf_dict = {}
        for page in os.listdir("temp"):
            parsed = parser.from_file(os.path.join('temp', page))
            text = parsed["content"]
            payload = {"text": text}

            r = requests.post(url, data=payload)
            temp_dict = None
            if r.status_code == requests.codes.ok:
                annotations = r.json()["measurements"]
                #print(type(annotations))
                test = EntityLabeler(text, annotations, unit_entities)

                temp_dict = test.parse()
            else:
                test = EntityLabeler(text, None, unit_entities)
                temp_dict = test.parse_no_annot()

            if temp_dict:
                temp_dict["page"] = page
            df = df.append(pd.DataFrame(temp_dict))


        df.to_csv(f, index=False)
        print("Writing results to ", outpath)
        key_counts = df.count().to_dict()

        shutil.rmtree("temp")
        return key_counts

if __name__ == "__main__":
    keyDict = {}
    for file in os.listdir("../running"):
        out = file + "_output.csv"
        fpath = os.path.join("../running", file)
        resultDict = run_entitylabeler(fpath, out)

        print("Number of Entities Found: %d" % (len(resultDict.keys())))

        keyDict = Counter(keyDict) + Counter(resultDict)
        print("Total Number of Entities Found: %d" % (len(keyDict.keys())))
        print(len(keyDict.keys()), sum(keyDict.values()))
    jsonDict = json.dumps(keyDict)
    with open ("%s_Dict_GQ.json" % (demo_files), 'w') as file:
        file.write(jsonDict)
