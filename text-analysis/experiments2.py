import spacy
#import en_core_web_sm
from nltk import ngrams
import collections
import string
import tika
tika.initVM()
import re
from tika import parser
import pandas as pd
import PyPDF2
import os
import shutil
import ast
import numpy as np
import dill
import click
import spacy

# ========= Data structures, initializations and hyperparameters

global PREP, PUNC, WORD, DIGI, UNIT
global prepos, punc, units
global threshold, current_document, counter
global learned_patterns, all_patterns, current_patterns, interesting_patterns, fuzzy_patterns

def split_pdf_pages(input_pdf_path, target_dir, fname_fmt=u"{num_page:04d}.pdf"):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if 'doc' in input_pdf_path:
        shutil.copyfile(input_pdf_path, (target_dir + "/delete"))
        return

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


def parse_document(file_path):

    global current_document
    current_document=file_path.split('/')[-1]

    parsed_text=[]
    # create a dir for dumping split pdfs
    if os.path.exists('./temp'):
        shutil.rmtree('./temp/')
    else:
        os.mkdir('./temp')
    split_pdf_pages(file_path, 'temp')

    for pdf_page in os.listdir('temp'):
        # print('processing page: ',pdf_page)
        parsed = parser.from_file(os.path.join('temp', pdf_page))
        try:
            pdftext = parsed['content']
        except Exception:
            print("Could not read file.")
            pdftext=''

        parsed_text.append(pdftext)
    parsed_text = " ".join(parsed_text)

    return parsed_text.lower()



nlp = spacy.load('en_core_web_sm')
doc = parse_document('../.')
print(doc[0:10000])
tokenized = nlp(doc)

for token in tokenized:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)