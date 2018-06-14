# -*- coding: utf-8 -*-

from nltk import ngrams
import collections
import string
import subprocess
import tika
tika.initVM()
import re
from tika import parser
import pandas as pd
import PyPDF2
import os
import shutil
import sys
# Assumptions:
    # 1. The related entities are very close to each other in space (adjacent)
    # 2. The entities repeat to give rise statistical significance
    # 3. The relationships follow loosly some underlying rule

# Research wise interesting questions:
    # 1. How well can these rules be learned using supervision ?

# Future things:
    # 1. Make things more generalized: better Auto-Tokenization-signs, add more entity recoznition to create tokens-signs
    # 2. Smarter pattern merging
    # 3. Use also dependency parsing to create token-signs and rules
    # 4. Make it easier to filter patterns, add general patterns whitelisting rules
    # 5. Rank also based on more 'congruence' or patterns (since the text is so un-sentenced), eg. sentence congruence detection
    # 5. Add better/NLP based tokenization and chunking
    # 6. Investigate the recall

def createsign(string):
    def decide(arr):
        if arr[0]=='A' and len(arr)>0:
            return 'A'
        if arr[0]=='d' and len(arr)>0:
            return 'D~'
        else:
            return ''.join(arr)

    signature=''
    string=str(string)
    for ch in string:
        if ch.isalpha():
            signature+='A'
        elif ch.isnumeric():
            signature+='d'
        else:
            signature+=ch

    # condense the patterns to mark words
    condensed_signature=''
    buffer=[]
    for i, ch in enumerate(signature):
        buffer.append(ch)
        if i!=len(signature)-1 and ch!=signature[i+1]:
            condensed_signature+=decide(buffer)
            buffer=[]
        elif i==len(signature)-1:
            condensed_signature+=decide(buffer)


    return condensed_signature



def split_pdf_pages(input_pdf_path, target_dir, fname_fmt=u"{num_page:04d}.pdf"):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if 'doc' in input_pdf_path:
        shutil.copyfile(input_pdf_path, (target_dir + "/delete"))
        return

    with open(input_pdf_path, "rb") as input_stream:
        try:
            input_pdf = PyPDF2.PdfFileReader(input_stream)
        except PyPDF2.utils.PdfReadError:
            print("File " + input_pdf_path + "could not be read. EOF marker not found")
            return

        if input_pdf.flattenedPages is None:
            # flatten the file using getNumPages()
            input_pdf.getNumPages()  # or call input_pdf._flatten()

        for num_page, page in enumerate(input_pdf.flattenedPages):
            output = PyPDF2.PdfFileWriter()
            output.addPage(page)

            file_name = os.path.join(target_dir, fname_fmt.format(num_page=num_page))
            with open(file_name, "wb") as output_stream:
                output.write(output_stream)

def create_csv(filepath):
    pdf_buffer={}

    # df=pd.DataFrame()
    # df=df.append({"A":1,"B":2},ignore_index=True)
    # df=df.append({"A":1,"C":2},ignore_index=True)
    # print(df)
    # exit()


    # create a dir for dumping split pdfs
    if os.path.isdir('./temp'):
        shutil.rmtree('./temp')
    os.mkdir('./temp')

    split_pdf_pages(filepath,'temp')
    
    # parsed={}
    # parsed["content"]='hello veejbd jbd hjhj, adjb 898.98   q989fff.'

    all_rules=[]
    all_instances=[]
    rule_to_instance={} # rule:line/positional-id::instance
    for pdf_page in os.listdir('temp'):
        # print('processing page: ',pdf_page)
        parsed = parser.from_file(os.path.join('temp',pdf_page))
       # parsed = subprocess.call("tika --config=tika_config.xml " + fpath)
        try:
            pdftext=parsed['content']
            print(pdftext)
        # TODO Find more specific excepetion.
        except Exception:
            print("Could not read file.")
            continue
        page_rules=[]
        page_instances=[]
        for line in pdftext.split('\n'):
            # clean
            # create chunks by dividing on commas+space, period+space and multi-space so that patterns don't span beyond them
            # chunks=re.split(', |\. |\s{2,}',line)
            chunks = re.split(', |\. ', line)
            # print(line,chunks)
            # remove commas from numbers (8,643), give valid spacing around #, = and @
            # tokenize everything based on spaces/tabs
            chunks=[chunk.replace(",","").replace("="," = ").replace("@"," @ ").replace("#"," # ").replace("$"," $ ").
                        replace("°"," ° ").replace("%"," % ").replace("\""," \" ").replace("'"," ' ").replace(":"," : ").split() for chunk in chunks]
            # convert each into a (semi)signature: first to a signature, then revert the words back to life
            chunks_signed=[list(map(createsign,chunk)) for chunk in chunks]


            for i, chunk in enumerate(chunks_signed):
                for j,token in enumerate(chunk):
                    if 'A' in token:
                        chunks_signed[i][j]=chunks[i][j]

            # create n-grams

            n_gram_range=(3,4,5,6,7)
            for n in n_gram_range:
                filtered_grams_signed=[]
                filtered_grams= []
                all_grams_signed=list(map(lambda x: list(ngrams(x,n)),chunks_signed))
                all_grams = list(map(lambda x: list(ngrams(x, n)), chunks))
                # flatten the nested list
                all_grams_signed=[item for sublist in all_grams_signed for item in sublist]
                all_grams = [item for sublist in all_grams for item in sublist]

                for gram_signed, gram in zip(all_grams_signed,all_grams):
                    # remove ngrams that are all alphabets
                    flag = 0
                    for token in gram_signed:
                        if 'D~' in token:
                            flag=1
                            break
                    if flag==1:
                        filtered_grams_signed.append(gram_signed)
                        filtered_grams.append(gram)

                page_rules.extend(filtered_grams_signed)
                page_instances.extend(filtered_grams)

        all_rules.append(page_rules)
        all_instances.append(page_instances)


    # find statistically significant n-grams (first flatten the rule list)
    counted=collections.Counter([item for sublist in all_rules for item in sublist])
    # significant=counted.most_common(100)


    # get the longest pattern with the same support
    filtered_patterns={}

    for pattern in counted.keys():
        # create the ngrams/subsets of a set and check if they are already present, if so check minsup and delete
        len_pattern=len(pattern)
        filtered_patterns[pattern]=counted[pattern]
        for i in range(1,len_pattern):
            subpatterns=list(ngrams(pattern,i))
            for subpattern in subpatterns:
                if subpattern in filtered_patterns.keys() and filtered_patterns[subpattern]==counted[pattern]:
                    # delete subpattern
                    # print('deleting',subpattern,', because',pattern,filtered_pattens[subpattern],counted[pattern])
                    filtered_patterns.pop(subpattern)

    # print(len(filtered_pattens))
    # print(filtered_pattens)

    # remove that have minsup of less than threshold

    for pattern in list(filtered_patterns):
        if filtered_patterns[pattern]<10: # todo: do df(x)/dt and find an auto cut-off
            filtered_patterns.pop(pattern)



    # print(len(filtered_pattens))
    # print(filtered_pattens)

    filtered_patterns=list(filtered_patterns)

    # merge the windowed patterns (adjacent only)
    # refiltered_patterns=[]
    # buffer=[]
    # for i, pattern in enumerate(filtered_patterns):
    #     if i==0 :
    #         buffer=list(pattern)
    #         continue
    #
    #     if pattern[:-1]==list(filtered_patterns)[i-1][1:]:
    #         buffer.append(pattern[-1])
    #     else:
    #         refiltered_patterns.append(tuple(buffer))
    #         buffer=list(pattern)
    #
    # # transfer the last remaining things from the buffer
    # refiltered_patterns.append(tuple(buffer))
    # filtered_patterns=refiltered_patterns



    # filter out patterns that start with a preposition, a number or a punctuation
    prepos=['aboard','about','above','across','after','against','along','amid','among','anti','around','as',
            'at','before','behind','below','beneath','beside','besides','between','beyond','but','by',
            'concerning','considering','despite','down','during','except','excepting','excluding','following',
            'for','from','in','inside','into','like','minus','near','of','off','on','onto','opposite','outside',
            'over','past','per','plus','regarding','round','save','since','than','through','to','toward','towards',
            'under','underneath','unlike','until','up','upon','versus','via','with','within','without']

    punc=set(string.punctuation)

    filtered_patterns=list(filter(lambda x:x[0].lower() not in prepos and 'D~' not in x[0] and x[0][0] not in punc, filtered_patterns))


    print(len(filtered_patterns))
    print(filtered_patterns)

    # separate records based on date and periodicity of patterns and add things to csv
        # for now just add them one by one

    units=['ft','gal','ppa','psi','lbs','lb','bpm','bbls','bbl','\'',"\"","'","°","$",'hrs']

    df=pd.DataFrame()



    for pagenum, page_rules, page_instances in zip(range(len(all_rules)),all_rules,all_instances):
        row_buffer = {}
        for pattern, instance in zip(page_rules, page_instances):
            row_buffer['page']=[str(pagenum)]
            if pattern in filtered_patterns:
                # separate the entity, connector, value, unit etc.
                # todo: here a better analysis of the sentence structure may play a part in dividing the entity a bit further
                entity=[]
                value=[]
                unit=[]
                for sign,token in zip(pattern,instance):
                    if sign.lower() in units:
                        unit.append(token)
                    elif "D~" in sign:
                        value.append(token)
                    elif sign.lower() in punc:
                        pass
                    else:
                        entity.append(token)


                column_name=' '.join(entity) + ' (' + ', '.join(unit) + ')' if len(unit)>0 else ' '.join(entity)
                cell=', '.join(value)
                if column_name not in row_buffer.keys():
                    row_buffer[column_name]=[]
                if column_name not in pdf_buffer.keys():
                    # for counting total entities
                    pdf_buffer[column_name]=0

                row_buffer[column_name].append(cell)
                pdf_buffer[column_name]+=1


        # beautify the list in every cell

        row_buffer={key : '\n'.join(value) for key, value in row_buffer.items()}
        row_buffer['vendor']=filepath.split('/')[-2]
        row_buffer['well']=filepath.split('/')[-1].split('.')[0]
        df=df.append(row_buffer,ignore_index=True)


    # df.to_csv('report.csv')
    shutil.rmtree('temp')

    return pdf_buffer, df

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[-1]
        print("Extracting %s..." % (filename))
    else:
        print("No file specified, using hardcoded path.")
        ### TODO ADD A PATH IN A FILE
        filename = '/YOUR_FILE/YOUR_PDF'
    pdf_buffer=create_csv(filename)
    print(pdf_buffer)


