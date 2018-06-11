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
import jellyfish
from fuzzywuzzy import fuzz
import dill
import click
import spacy


# ========= Data structures, initializations and hyperparameters

global PREP, PUNC, WORD, DIGI, UNIT
global prepos, punc, units
global threshold, current_document, counter
global learned_patterns, all_patterns, current_patterns, interesting_patterns, fuzzy_patterns
global nlp

nlp = spacy.load("en")

PREP='Prep~'
PUNC='Punc~'
WORD='Word~'
DIGI='Digi~'
UNIT='Unit~'

prepos = ['aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'anti', 'around',
              'as',
              'at', 'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by',
              'concerning', 'considering', 'despite', 'down', 'during', 'except', 'excepting', 'excluding', 'following',
              'for', 'from', 'in', 'inside', 'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto', 'opposite',
              'outside',
              'over', 'past', 'per', 'plus', 'regarding', 'round', 'save', 'since', 'than', 'through', 'to', 'toward',
              'towards',
              'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via', 'with', 'within', 'without',
              'and', 'or']
units = ['ft', 'gal', 'ppa', 'psi', 'lbs', 'lb', 'bpm', 'bbls', 'bbl', '\'', "\"", "'", "°", "$", 'hrs']
punc = set(string.punctuation)

# ========== utility functions

def remove_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

def savemodel(model,outfile):
    with open(outfile, 'wb') as output:
        dill.dump(model, output)
    return ''

def loadmodel(infile):
    model=''
    with open(infile, 'rb') as inp:
        model = dill.load(inp)
    return model

def ispunc(string):
    if re.match('[^a-zA-Z\d]',string):
        return True
    return False

# def break_natural_boundaries(text):
#     spacy_text = nlp(text)
#     tokenized = [str(token) for token in spacy_text]
#     print(tokenized)
#     return(tokenized)

def break_natural_boundaries(string):

    stringbreak=[]
    if len(string.split(' ')) > 1:
        stringbreak = string.split(' ')
    else:
        # spl = '[\.\,|\%|\$|\^|\*|\@|\!|\_|\-|\(|\)|\:|\;|\'|\"|\{|\}|\[|\]|]'
        alpha = '[A-z]'
        num = '\d'
        spl='[^A-z\d]'



        matchindex = set()
        matchindex.update(set(m.start() for m in re.finditer(num + alpha, string)))
        matchindex.update(set(m.start() for m in re.finditer(alpha + num, string)))
        matchindex.update(set(m.start() for m in re.finditer(spl + alpha, string)))
        matchindex.update(set(m.start() for m in re.finditer(alpha + spl, string)))
        matchindex.update(set(m.start() for m in re.finditer(spl + num, string)))
        matchindex.update(set(m.start() for m in re.finditer(num + spl, string)))
        matchindex.update(set(m.start() for m in re.finditer(spl + spl, string)))

        matchindex.add(len(string)-1)
        matchindex = sorted(matchindex)
        start = 0

        for i in matchindex:
            end = i
            stringbreak.append(string[start:end + 1])
            start = i+1
    stringbreak = [s for s in stringbreak if s]
    return stringbreak

def break_and_split(arr):
    new_arr=[]
    for token in arr:
        new_arr.extend(break_natural_boundaries(token))
    return new_arr

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

def levenshtein_similarity(s, t):
    """ Levenshtein Similarity """
    Ns = len(s);
    Nt = len(t);
    lev_sim = 1.0 - (jellyfish.levenshtein_distance(s, t)) / float(max(Ns, Nt))
    return lev_sim

def word_similarity(s,t, type=''):
    if type=='leven':
        return levenshtein_similarity(s, t)
    else:
        return float(fuzz.ratio(s.upper(), t.upper()))/100


# ========== state changing functions

def find_entites(hpattern, mask=[]):
    '''
    aggrerate the words that are next to each other as an entity. Find multiple ones.
    :param hpattern:
    :return:
    '''

    if len(mask) == 0:
        mask = list(np.full(len(hpattern), True))

    entities=[]
    entity=''
    dummied_hpatteren=list(hpattern)
    dummied_hpatteren.append(('~', '~', '~'))
    dummied_hpatteren=tuple(dummied_hpatteren)
    mask.append(True)

    for token, select in zip(dummied_hpatteren, mask):
        if not select:
            continue
        if token[2]==WORD:
            entity+=' '+token[0]
        else:
            if entity!='':
                entities.append(entity)
            entity = ''
    return entities

def find_units(hpattern, mask=[]):
    '''
    find the units in the pattern
    :param hpattern:
    :return:
    '''

    if len(mask) == 0:
        mask = list(np.full(len(hpattern), True))

    units=[]
    for token, select in zip(hpattern,mask):
        if not select:
            continue
        if len(token)>=4 and token[3]==UNIT:
            units.append(token[0])
    return units

def find_values(instance, hpattern, mask=[]):
    '''
    find the values in the pattern
    :param hpattern:
    :return:
    '''

    values=[]
    if len(mask)==0:
        mask=list(np.full(len(hpattern),True))

    for token_inst,token_patt,select in zip(instance, hpattern, mask):
        if not select:
            continue
        if token_patt[2]==DIGI :
            values.append(token_inst)
    return values

def find_exact_patterns(hpattern):
    '''
    finds the hpatterns that are exact to the given hpattern
    look by base patterns, as they don't have the variable/value
    :param hpattern:
    :return:
    '''
    global current_patterns
    exact_pattern_ids=[]
    base_pattern=str(get_base_pattern(ast.literal_eval(hpattern)))
    if base_pattern in list(current_patterns['base_pattern']):
        exact_pattern_ids.append(list(current_patterns[current_patterns['base_pattern']==base_pattern]['pattern_id'])[0])
    return exact_pattern_ids

def find_close_patterns(hpattern):
    '''
    finds the hpatterns that are closest to the given hpattern
    :param hpattern:
    :return:
    '''
    global current_patterns
    close_pattern_ids=[]

    hpattern=ast.literal_eval(hpattern)
    entities=find_entites(hpattern)
    units=find_units(hpattern)


    close_patterns=[]

    for _, row in current_patterns.iterrows():
        confidence_flag_entity = 0
        confidence_flag_unit = 0
        confidence=0 # todo: give the best score here; will help decide the rank
        hpattern_iter=ast.literal_eval(str(row['hpattern']))
        mask = str(row['mask'])
        if mask == '':
            mask = []
        else:
            mask = ast.literal_eval(str(row['mask']))
        entities_iter=find_entites(hpattern_iter,mask)
        units_iter=find_units(hpattern_iter,mask)

        for entity_iter in entities_iter:
            for entity in entities:
                if word_similarity(entity,entity_iter)>0.7:
                    confidence_flag_entity=1

        for unit_iter in units_iter:
            for unit in units:
                if unit.lower()==unit_iter.lower():
                    confidence_flag_unit=1

        if confidence_flag_entity==1 or confidence_flag_unit==1:
            close_patterns.append((row['pattern_id'],confidence_flag_entity,confidence_flag_unit))



    # todo: here rank the patterns according to confidence and return the top n

    for conf in close_patterns:
        close_pattern_ids.append(conf[0])


    return close_pattern_ids

def find_far_patterns(entity_name, aliases=[]):
    '''
    finds the patterns that have similar entity names
    :param entity_name:
    :return:
    '''
    global current_patterns
    far_pattern_ids=[]

    aliases.append(entity_name)
    for _, row in current_patterns.iterrows():

        mask = str(row['mask'])
        if mask == '':
            mask = []
        else:
            mask=ast.literal_eval(str(row['mask']))

        hpattern_iter = ast.literal_eval(str(row['hpattern']))
        entities_iter = find_entites(hpattern_iter, mask)

        for entity_iter in entities_iter:
            for alias in aliases:
                if word_similarity(alias, entity_iter) > 0.5:
                    far_pattern_ids.append(row['pattern_id'])


    return far_pattern_ids

def matcher_bo_entity(entity_name):
    '''
    if the entity name is already present in the learned_patterns, it gets the exact pattern. Then checks if it is present in the current_patterns.
    if present then just returns the exact pattern. If not, then finds the closest pattern in current_pattern.

    :param entity_name:
    :return:
    '''

    global learned_patterns
    global all_patterns

    pre_learned_patterns=[]
    pre_learned_masks=[]
    seed_aliases=[]
    exact_pattern_ids=[]
    exact_masks = {}
    close_pattern_ids=[]
    far_pattern_ids=[]


    # check if the any patterns for the entity have already been identified
    if entity_name in list(learned_patterns['entity_name']):
        seed_aliases=str(list(learned_patterns[learned_patterns['entity_name'] == entity_name]['seed_aliases'])[0])
        seed_aliases=seed_aliases.split(',')
        pattern_ids=str(list(learned_patterns[learned_patterns['entity_name'] == entity_name]['pattern_ids'])[0])
        if pattern_ids!='':
            pattern_ids=ast.literal_eval(pattern_ids)
            for pattern_id in pattern_ids:
                # get the pattern using the id
                pre_learned_patterns.append(str(list(all_patterns[all_patterns['pattern_id']==pattern_id]['hpattern'])[0]))
                pre_learned_mask=str(list(all_patterns[all_patterns['pattern_id'] == pattern_id]['mask'])[0])
                if pre_learned_mask!='':
                    pre_learned_masks.append(ast.literal_eval(pre_learned_mask))
                else:
                    pre_learned_masks.append([])



    # find suitable current patterns
    if len(pre_learned_patterns)!=0:
        print('We have seen this entity before! Let us find if the exact patterns work...')
        for hpattern, mask in zip(pre_learned_patterns, pre_learned_masks):
            # check if the exact pattern is present in the current patterns
            exact_hpatterns_found=find_exact_patterns(hpattern)
            exact_pattern_ids.extend(exact_hpatterns_found)
            for pattern_id in exact_hpatterns_found:
                exact_masks[pattern_id]=mask

        if len(exact_pattern_ids)>0:
            print('looks like the entity is present in the same form! Great!')

        else:
            print('finding patterns closer to learned patterns ...')
            for hpattern in pre_learned_patterns:
                # find the closest patterns
                close_pattern_ids.extend(find_close_patterns(hpattern))

    else:
        # find the patterns that have similar entity name
        print('looks like nothing is close enough is there! Let us just find the closest seeming entity by the name!')
        far_pattern_ids.extend(find_far_patterns(entity_name,aliases=seed_aliases))

    return exact_pattern_ids, close_pattern_ids, far_pattern_ids, exact_masks

def matcher_bo_value(entity_value):
    '''
    searches for all the patterns in current_pattern that have the particular value associated with them
    :param entity_value:
    :return:
    '''
    global current_patterns
    exact_pattern_ids=[]
    instance_samples=[] # one instance per pattern

    for _, row in current_patterns.iterrows():

        instances=ast.literal_eval(str(row['instances']))
        for instance in instances:
            if entity_value in instance:
                exact_pattern_ids.append(row['pattern_id'])
                instance_samples.append(instance)
                break

    return exact_pattern_ids, instance_samples

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

    return parsed_text

def read_seed_matchings(path):
    '''
    read the user provided 'entity_name' : 'seed_aliases' : 'learned_aliases' and initialize the learned_patterns with it
    :param path:
    :return:
    '''
    global learned_patterns

def filter1(row):
    '''
    Returns True if the pattern satisfies a certain criteria, else False
    :param row:
    :return:
    '''
    global threshold

    # if the pattern occurs in the document less than the threshold then return false
    if int(row['num_instances'])>threshold:
        return True
    return False

def filter2(row):
    '''
    Returns True if the pattern satisfies a certain criteria, else False
    :param row:
    :return:
    '''

    pattern=ast.literal_eval(str(row['hpattern']))
    # if the first token is preposition/pronoun or punctuation then return false
    if pattern[0][2] ==PREP or pattern[0][2] ==PUNC:
        return False

    return True

def filter3(row):
    '''
    Returns True if the pattern satisfies a certain criteria, else False
    :param row:
    :return:
    '''

    pattern=ast.literal_eval(str(row['hpattern']))
    for token in pattern:
        # if atleast one entity/unit found, it is okay
        if token[2] == WORD:
            return True
    return False

def filter4(row):
    '''
    Returns True if the pattern satisfies a certain criteria, else False
    :param row:
    :return:
    '''

    pattern=ast.literal_eval(str(row['hpattern']))
    for token in pattern:
        # if atleast one number found, it is okay
        if token[2] == DIGI:
            return True
    return False

def apply_filters(fltr):
    '''
    Apply filters to remove 'irrelevant' current patterns: see filter1 impl
    :param: filter: a function
    :return:
    '''

    global current_patterns

    current_patterns=current_patterns[current_patterns.apply(lambda x: fltr(x), axis=1)]
    print('FILTERED! now number of patterns: ', len(current_patterns))

def getID():
    global counter
    counter+=1
    return counter

def get_base_pattern(hpattern):
    '''
    takes the second level of an hpattern (non variable tokens)
    :param hpattern:
    :return:
    '''
    base_pattern=[]
    for patt in hpattern:
        base_pattern.append(patt[1])

    return tuple(base_pattern)

def create_hpattern(instance):
    '''
    creates a heirarchy of 'denominations/classes' for each base pattern
    :param instance:
    :return: base_pattern, h_pattern
    '''
    global punc
    global prepos
    global units

    signature = []

    for token in instance:
        if not token:
            continue
        if token in prepos:
            signature.append((token, token, PREP))
        elif token.isnumeric():
            signature.append((token, DIGI, DIGI))
        elif token.isalpha():
            sign=[token, token, WORD]
            if token.lower() in units:
                sign.append(UNIT)
            signature.append(tuple(sign))
        elif ispunc(token):
            signature.append((token, token, PUNC))
        else:
            signature.append((token))

    return tuple(signature)

def create_patterns_per_doc(parsed_text):
    '''

    :param parsed_text: it should be a list of texts. One item/text for every page in the document.
    :return:
    '''
    global current_patterns
    global current_document_path
    global all_patterns

    all_hpatterns=[]
    all_base_patterns=[]
    all_instances = []

    for page in parsed_text:
        page_hpatterns=[]
        page_base_patterns=[]
        page_instances = []
        for line in page.split('\n'):  # pattern analysis is done based on each line
            if not line:
                continue
            # create chunks by dividing on commas+space, period+space (and multi-space??) so that patterns don't span beyond them
            # chunks=re.split(', |\. |\s{2,}',line)
            chunks = re.split(', |\. |;', line.lower())
            print(line,chunks)

            # remove commas from numbers (8,643), give valid spacing around #, = and @
            # tokenize everything based on spaces/tabs
            # creates a list(token) of lists(chunk): [[token,token,token],[token,token]]
            # chunks = [
            #     chunk.replace(",", "").replace("=", " = ").replace("@", " @ ").replace("#", " # ").replace("$", " $ ").
            #         replace("°", " ° ").replace("%", " % ").replace("\"", " \" ").replace("'", " ' ").replace(":",
            #                                                                                                   " : ").split()
            #     for chunk in chunks]
            chunks = [re.sub(r"([^a-zA-Z0-9])", r" \1 ", chunk.replace(",", "")) for chunk in chunks]

            # separate the tokens further using the natural sepamration boundaries
            # chunks = [break_and_split(chunk) for chunk in chunks]
            chunks = [break_natural_boundaries(chunk) for chunk in chunks]
            chunks_base_patterns=[]
            chunks_hpatterns=[]

            for chunk in chunks:
            # convert each chunk to base pattern and hpattern
                if not chunk:
                    continue
                print(chunk)
                print(type(chunk))
                print(type(chunk[0]))
                hpattern=create_hpattern(chunk)
                base_pattern=get_base_pattern(hpattern)
                chunks_base_patterns.append(base_pattern)
                chunks_hpatterns.append(hpattern)

            print("chunks", chunks)
            print("chunks_base_patterns", chunks_base_patterns)
            print("chunks_hpatterns", chunks_hpatterns)
            # create n-grams

            n_gram_range = (3, 4, 5, 6, 7)
            for n in n_gram_range:

                all_grams_base_patterns = list(map(lambda x: list(ngrams(x, n)), chunks_base_patterns))
                all_grams_hpatterns = list(map(lambda x: list(ngrams(x, n)), chunks_hpatterns))
                all_grams = list(map(lambda x: list(ngrams(x, n)), chunks))

                # flatten the nested list
                all_grams_base_patterns = [item for sublist in all_grams_base_patterns for item in sublist]
                print("all_grams_base_patterns", all_grams_base_patterns)
                all_grams_hpatterns = [item for sublist in all_grams_hpatterns for item in sublist]
                print("all_grams_hpatterns", all_grams_hpatterns)
                all_grams = [item for sublist in all_grams for item in sublist]
                print("all_grams", all_grams)

                page_base_patterns.extend(all_grams_base_patterns)
                page_hpatterns.extend(all_grams_hpatterns)
                page_instances.extend(all_grams)


        all_base_patterns.append(page_base_patterns)
        all_hpatterns.append(page_hpatterns)
        all_instances.append(page_instances)
    print("all base patterns", all_base_patterns)
    all_page_numbers=[]
    for indx, _ in enumerate(all_instances):
        all_page_numbers.append(list(np.full(len(_),indx+1)))

    all_base_patterns_flattened=[item for sublist in all_base_patterns for item in sublist]
    print("all base patterns flattened", all_base_patterns_flattened)
    all_hpatterns_flattened = [item for sublist in all_hpatterns for item in sublist]
    all_instances_flattened = [item for sublist in all_instances for item in sublist]
    all_page_numbers_flattened=[item for sublist in all_page_numbers for item in sublist]

    counted_patterns = collections.Counter(all_base_patterns_flattened)
    print(counted_patterns)
    exit()
    # ======= get the longest pattern with the same support (keeps only the superset, based on minsup criteria)
    # todo: check if works correctly
    filtered_patterns = {}
    for pattern in counted_patterns.keys():
        # create the ngrams/subsets of a set and check if they are already present, if so check minsup and delete
        len_pattern = len(pattern)
        filtered_patterns[pattern] = counted_patterns[pattern]
        for i in range(1, len_pattern):
            # create all size sub patterns/n-grams
            subpatterns = list(ngrams(pattern, i))
            for subpattern in subpatterns:
                if subpattern in filtered_patterns.keys() and filtered_patterns[subpattern] == counted_patterns[pattern]:
                    # delete subpattern
                    # print('deleting',subpattern,', because: ', pattern, filtered_pattens[subpattern], counted[pattern])
                    filtered_patterns.pop(subpattern)

    # ========== create data frame

    # aggregate the instances based on base patterns
    # create a mapping from base pattern to hpattern
    aggregated_pattern_instance_mapping={}
    aggregated_pattern_pagenumber_mapping={}
    base_pattern_to_hpattern={}
    for pattern, hpattern, instance, page_number in zip(all_base_patterns_flattened, all_hpatterns_flattened, all_instances_flattened,all_page_numbers_flattened):

        # aggregate
        if pattern not in aggregated_pattern_instance_mapping.keys():
            aggregated_pattern_instance_mapping[pattern]=[]
            aggregated_pattern_pagenumber_mapping[pattern]=[]

        aggregated_pattern_instance_mapping[pattern].append(instance)
        aggregated_pattern_pagenumber_mapping[pattern].append(page_number)

        # mapping
        if pattern not in base_pattern_to_hpattern.keys():
            base_pattern_to_hpattern[pattern]=hpattern


    for pattern in aggregated_pattern_instance_mapping.keys():
        if pattern in filtered_patterns:
            pattern_id=getID()

            current_patterns=current_patterns.append({'pattern_id':pattern_id,'base_pattern':str(pattern),'instances':str(aggregated_pattern_instance_mapping[pattern]),
                                 'page_numbers':aggregated_pattern_pagenumber_mapping[pattern],'hpattern':str(base_pattern_to_hpattern[pattern]),'document_name':current_document,'num_instances':counted_patterns[pattern]}, ignore_index=True)




    # ============= apply filters

    # filter the patterns that have the number of instances below a certain threshold
    apply_filters(filter1)
    # remove the ones that start with a punctuation or preposition
    apply_filters(filter2)
    # remove the patterns that have only punctuations, prepositions and numbers
    apply_filters(filter3)
    # remove the ones that have no numbers
    apply_filters(filter4)

    current_patterns = current_patterns.replace(np.nan, '', regex=True)
    current_patterns.to_csv('current_patterns.csv')

def find_interesting_patterns():
    '''
    using the list of other patterns, find the matching patterns from the current document
    :param patterns:
    :return:
    '''
    global interesting_patterns

def init(file_path, fresh=False):
    '''
    initialize and load all the relevant dataframes and datastructures
    param:
    file_path
    fresh : if True then initialize everything anew
    :return:
    '''

    global prepos, punc, units
    global threshold, current_document_path, counter
    global learned_patterns, all_patterns, current_patterns, other_patterns, other_pattern_instances

    # prepos = ['aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'anti', 'around',
    #           'as',
    #           'at', 'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by',
    #           'concerning', 'considering', 'despite', 'down', 'during', 'except', 'excepting', 'excluding', 'following',
    #           'for', 'from', 'in', 'inside', 'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto', 'opposite',
    #           'outside',
    #           'over', 'past', 'per', 'plus', 'regarding', 'round', 'save', 'since', 'than', 'through', 'to', 'toward',
    #           'towards',
    #           'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via', 'with', 'within', 'without',
    #           'and', 'or']
    # units = ['ft', 'gal', 'ppa', 'psi', 'lbs', 'lb', 'bpm', 'bbls', 'bbl', '\'', "\"", "'", "°", "$", 'hrs']
    # punc = set(string.punctuation)

    threshold = 10
    # save state across documents
    if os.path.exists('counter'):
        counter=loadmodel('counter')
    else:
        counter = 0
    print('counter',counter)
    current_document_path = ''

    # entity matchings for all the documents processed so far
    if os.path.exists('learned_patterns.csv'):
        learned_patterns = pd.read_csv('learned_patterns.csv', index_col=0)
        learned_patterns = learned_patterns.replace(np.nan, '', regex=True)
    else:
        learned_patterns = pd.DataFrame(columns=['entity_name', 'seed_aliases', 'pattern_ids'])

    # pattern information about all the patterns seen so far from all the documents processed
    if os.path.exists('all_patterns.csv'):
        all_patterns = pd.read_csv('all_patterns.csv', index_col=0)
        all_patterns = all_patterns.replace(np.nan, '', regex=True)
    else:
        all_patterns = pd.DataFrame(
            columns=['pattern_id', 'base_pattern', 'instances', 'hpattern', 'document_name', 'num_instances', 'mask','page_numbers'])

    # pattern information about just the current pdf/document
    current_patterns = pd.DataFrame(
        columns=['pattern_id', 'base_pattern', 'instances', 'hpattern', 'document_name', 'num_instnces', 'mask','page_numbers'])

    other_patterns = pd.DataFrame(columns=['hpattern', 'instances', 'document_name'])

    parsed_text = parse_document(file_path)

    print('Creating patterns for the document: '+file_path)
    create_patterns_per_doc(parsed_text)

    # todo: remove this, just for testing, uncomment above
    # current_patterns= pd.read_csv('current_patterns.csv',index_col=0)
    # current_patterns = current_patterns.replace(np.nan, '', regex=True)

    all_patterns = pd.concat([all_patterns, current_patterns])

def close():
    '''

    :return:
    '''
    global all_patterns
    global current_patterns
    global counter

    # ============ add to the list of all patterns seen so far
    # all_patterns=pd.concat([all_patterns, current_patterns]) # done in init() now
    # all_patterns.to_csv('all_patterns.csv')
    savemodel(counter,'counter')
    build_report()
    # todo: add the finding of 'interesting' patterns


# ==================================== report building

def build_report():
    '''
    when the training is done, build the report based on the learned structures
    :return:
    '''
    global all_patterns
    global learned_patterns

    report=pd.DataFrame(columns=['page number','document name'])

    for index, row in learned_patterns.iterrows():
        entity_name=row['entity_name']
        pattern_ids=ast.literal_eval(str(row['pattern_ids']))

        for pattern_id in pattern_ids:
            report_temp=pd.DataFrame(columns=[entity_name,'page number','document name'])

            row=all_patterns[all_patterns['pattern_id']==pattern_id]

            instances=ast.literal_eval(str(list(row['instances'])[0]))
            hpattern=ast.literal_eval(str(list(row['hpattern'])[0]))
            mask=ast.literal_eval(str(list(row['mask'])[0]))
            if mask=='':
                mask=[]
            page_numbers = ast.literal_eval(str(list(row['page_numbers'])[0]))
            document_name=list(row['document_name'])[0]
            document_names=[document_name] * len(page_numbers)

            values_in_instances=[]
            for instance in instances:
                values=find_values(instance, hpattern, mask)
                values_in_instances.append(' '.join(values))

            report_temp[entity_name]=values_in_instances
            report_temp['page number']=page_numbers
            report_temp['document name']=document_names


            # aggregate by page number
            def agg_for_doc(series):
                return list(series)[0]
            def agg_for_entity_values(series):
                return '\n'.join(series)


            report_temp=report_temp.groupby(['page number'],as_index=False).agg({'document name': agg_for_doc,
                                                                entity_name: agg_for_entity_values,
                                                                })
            report=pd.merge(report, report_temp, how='outer', on=['page number', 'document name'])

            new_names = {}
            for col_name in list(report.columns):
                if '_x' in col_name or '_y' in col_name:
                    new_names[col_name] = col_name[:-2]
            report = report.rename(index=str, columns=new_names)

            def sjoin(x):
                return ';'.join(x[x.notnull()].astype(str))

            report = report.groupby(level=0, axis=1).apply(lambda x: x.apply(sjoin, axis=1))

    # aggregate by page number and document name one last time
    agg_dict = {}
    agg_func = lambda x: ' '.join(x)
    for column in report:
        if column != 'page number' and column != 'document name':
            agg_dict[column] = agg_func
    report =report.groupby(['page number', 'document name'],as_index=False).agg(agg_dict)
    report=report.sort_values(by=['document name','page number'])
    report.to_csv('report.csv')


# ==================================== CLI : command line functions

def get_rows_from_ids(ids):
    global current_patterns
    result_rows=current_patterns[current_patterns['pattern_id'].isin(ids)]

    return result_rows

def update_learn_pattern(entity_name, pattern_id, mask):
    '''
    based on user choice add a mask for the pattern
    1. update the pattern data structures for the masks
    2. add pattern id to the learning data structure
    :param pattern_id:
    :param choice:
    :return:
    '''

    global current_patterns
    global all_patterns
    global learned_patterns


    current_patterns.loc[current_patterns['pattern_id'] == pattern_id, 'mask'] = str(mask)
    all_patterns.loc[all_patterns['pattern_id'] == pattern_id, 'mask'] = str(mask)
    search = learned_patterns.loc[learned_patterns['entity_name'] == entity_name]
    if len(search) == 0:
        learned_patterns = learned_patterns.append({"entity_name": entity_name, "pattern_ids": [pattern_id], "seed_aliases":''},
                                                   ignore_index=True)
    else:
        pattern_ids = str(list(learned_patterns[learned_patterns['entity_name'] == entity_name]['pattern_ids'])[0])
        pattern_ids = ast.literal_eval(pattern_ids)
        pattern_ids.append(pattern_id)
        pattern_ids = str(list(set(pattern_ids)))
        learned_patterns.loc[learned_patterns['entity_name'] == entity_name, 'pattern_ids'] = pattern_ids

    learned_patterns.to_csv('learned_patterns.csv')
    current_patterns.to_csv('current_patterns.csv')
    all_patterns.to_csv('all_patterns.csv')

def interact_for_single_entity(entity_name, strict=False, auto_skip=False):
    '''
    1. look for exact/similar entities
    2. if strict is True and exact found, then update the learned
    3. Else, don't bother the user
    4. just skip that entity, put it in the report
    :param entity_name:
    :param strict:
    :return:
    '''

    result_rows = pd.DataFrame(columns=['instances', 'pattern_ids', 'hpattern'])
    exact_pattern_ids, close_pattern_ids, far_pattern_ids, exact_masks = matcher_bo_entity(entity_name)

    if len(exact_pattern_ids) != 0:
        result_rows = get_rows_from_ids(exact_pattern_ids)
    elif len(close_pattern_ids) != 0:
        result_rows = get_rows_from_ids(close_pattern_ids)
    elif len(far_pattern_ids) != 0:
        result_rows = get_rows_from_ids(far_pattern_ids)
    else:
        print('Looks like there is nothing to be found here!')

    if strict == True:
        if  len(exact_pattern_ids)==0:
            print('since no exact pattern found, and interactive is False. Skipping this entity.')
            return
        else :
            for _, row in result_rows.iterrows():
                pattern_id=row['pattern_id']
                mask=exact_masks[pattern_id]
                update_learn_pattern(entity_name, pattern_id, mask)

    else:
        if auto_skip==True and len(exact_pattern_ids)!=0:
            for _, row in result_rows.iterrows():
                pattern_id = row['pattern_id']
                mask = exact_masks[pattern_id]
                update_learn_pattern(entity_name, pattern_id, mask)
            return

        output = ''
        count = 0
        id_map = ['dummy']
        for _, row in result_rows.iterrows():
            count += 1
            output += str(count) + '. ' + ' '.join(ast.literal_eval(str(row['instances']))[0]) + '\n'
            id_map.append(row['pattern_id'])

        output+='ENTER s TO SKIP THIS ENTITY\n'
        output += 'ENTER v search the patterns by Value\n'

        print('Please, enter the index corresponding to the instance that best represents what you are looking for:')
        selected_pattern_id = input(output)

        if selected_pattern_id=='s':
            return

        if selected_pattern_id=='v':
            value=input('give the value for the entity: '+ entity_name)
            exact_pattern_ids, instance_samples=matcher_bo_value(value)
            output = ''
            count = 0
            id_map=['dummy']
            for pattern_id, instance in zip(exact_pattern_ids, instance_samples):
                count += 1
                output += str(count) + '. ' + ' '.join(instance) + '\n'
                id_map.append(pattern_id)

            output += 'ENTER s TO SKIP THIS ENTITY\n'
            if selected_pattern_id == 's':
                return

            print(
                'Please, enter the index corresponding to the instance that best represents what you are looking for:')
            selected_pattern_id = input(output)

            selected_instance=instance_samples[int(selected_pattern_id)-1]

        selected_instance = ast.literal_eval(
            str(list(result_rows[result_rows['pattern_id'] == id_map[int(selected_pattern_id)]]['instances'])[0]))[0]

        output = ''
        count = 0
        for token in selected_instance:
            count += 1
            output += str(count) + '. ' + token + '    '
        print(
            'Please, enter (seperated by spaces) all the indexes of the tokens that are relevant to the entity that you are looking for:')
        selected_tokens = input(output)
        selected_tokens = selected_tokens.strip().split(' ')
        selected_tokens = list(map(int,selected_tokens))
        mask = []
        for token_id in range(1, len(selected_instance)+1):
            if token_id in selected_tokens:
                mask.append(True)
            else:
                mask.append(False)

        update_learn_pattern(entity_name, id_map[int(selected_pattern_id)], mask)

def single_doc_cli(file_path, list_of_entity_names=[], strict=False, auto_skip=False):
    '''
    controls the flow of the CLI
    :param file_path:
    :return:
    '''

    init(file_path)

    if len(list_of_entity_names)>0:
         for entity_name in list_of_entity_names:
            print('Processing the entity: '+entity_name)
            interact_for_single_entity(entity_name, strict=strict, auto_skip=auto_skip)
         close()
    else:
        continue_cli = True
        while continue_cli:
            entity_name = input('Enter the entity you\'d like to search:')
            interact_for_single_entity(entity_name)
            decision=input('Do you want to continue searching: y or n')
            if decision=='n':
                continue_cli=False
                close()

@click.command()
@click.argument('path')
@click.option('--i/--no-i', default=True, help='choose --i for interactive')
@click.option('--a/--no-a', default=False, help='choose --a to auto skip exact patterns when in interactive mode. False by default.')
@click.option('--f/--no-f', default=False, help='choose --f to start fresh (all previous learning will be removed)')
@click.option('--e', help='give the path to the list of entities. One entity per line.')
def cli(path, i, a, f, e=''):
    '''
        detct if dir of file and accordingly iterates or not over multiple docs
        :param file_path:
        :return:
        '''

    if f:
        remove_files(['learned_patterns.csv','all_patterns.csv','current_patterns.csv','counter','report.csv'])

    strict=not i

    if e!='':
        list_of_entity_names=open(e,'r').readlines()
        list_of_entity_names=[name.strip() for name in list_of_entity_names]
    else:
        list_of_entity_names=[]

    import os
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file=='.DS_Store':
                    continue
                single_doc_cli(os.path.join(root, file), list_of_entity_names, strict, a)
    elif os.path.isfile(path):
        single_doc_cli(path, list_of_entity_names, strict, a)


# ============================================== begin the CLI

#cli()

# Example Commands:
# python knowledge_facilitator.py --i --e entity_name.txt '/path/to/the/pdf/name_of_the_pdf.pdf'
# python knowledge_facilitator.py --i --a --e entity_name.txt '/directory/to/the/pdfs'


# ====================================================== TODOS

# 9. do the interesting patterns
# 11. add more documentation of functions
# 12. add more logs
# 13. add an option to select all tokens
# 14. make it pip installable
# 15. caching by doc name
# 16. put a command line flag for just reports
# 17. when suggesting tokens to users, get rid of the punctuations (all punctuation within two selected things will be automatically true)
# 2. island the numbers together when outputting to the report (update the function find_values) : 80 . 6
# 4. instances to the user are not getting ranked (for later)





















# ============== example/test code

# print(ispunc(';;;;'))
# print(ispunc(';adad'))
# print(ispunc('adad'))

print(break_natural_boundaries('hello0098#%ji78'))
print(break_natural_boundaries("AVALANCHE 29-40 UNIT 1H/RICOCHET 29-40 UNIT 1H"))
print(break_natural_boundaries("ATP : 23 psi"))
print(break_and_split(['hello0098#%ji78']))
print(break_and_split(["AVALANCHE 29-40 UNIT 1H/RICOCHET 29-40 UNIT 1H"]))
print(break_and_split(["ATP : 23 psi"]))

# hpatterns=[]
# base_patterns=[]
# hpattern=create_hpattern(['ATP',':','23','psi'])
# print(hpattern)
# base_pattern=get_base_pattern(hpattern)
# print(base_pattern)
# print(list(ngrams(hpattern, 2)))
# print(list(ngrams(base_pattern, 2)))
# hpatterns.append(hpattern)
# base_patterns.append(base_pattern)
#
# hpattern=create_hpattern(['ATR',':','45','pli'])
# print(hpattern)
# base_pattern=get_base_pattern(hpattern)
# print(base_pattern)
# hpatterns.append(hpattern)
# base_patterns.append(base_pattern)
#
# print(base_patterns)
# print(hpatterns)

# parsed_text=parse_document('/path/something.pdf')
# create_patterns_per_doc(parsed_text)
# current_patterns=pd.read_csv('current_patterns.csv',index_col=0)
# current_patterns=current_patterns.replace(np.nan, '', regex=True)
# all_patterns=pd.read_csv('all_patterns.csv',index_col=0)
# all_patterns=all_patterns.replace(np.nan, '', regex=True)
# learned_patterns=pd.read_csv('learned_patterns.csv',index_col=0)
# learned_patterns=learned_patterns.replace(np.nan, '', regex=True)
# print(learned_patterns)
# matcher_bo_entity('Max pressure')
# exact_pattern_ids=find_exact_patterns("(('max', 'max', 'Word~'), ('rate', 'rate', 'Word~'), ('80', 'Digi~', 'Digi~'), ('.', '.', 'Punc~'), ('6', 'Digi~', 'Digi~'), ('bpm', 'bpm', 'Word~', 'Unit~'))")
# print(exact_pattern_ids)
# close_pattern_ids=find_close_patterns("(('Depth', 'Depth', 'Word~'), (':', ':', 'Punc~'), ('18750', 'Digi~', 'Digi~'))")
# print(close_pattern_ids)
# close_pattern_ids=find_close_patterns("(('CP', 'CP', 'Word~'), ('-', '-', 'Punc~'), ('0', 'Digi~', 'Digi~'), ('psi', 'psi', 'Word~', 'Unit~'))")
# print(close_pattern_ids)
# far_pattern_ids=find_far_patterns("Max")
# print(far_pattern_ids)
# exact_pattern_ids, samples=matcher_bo_value('635')
# print(exact_pattern_ids,samples)

# learned_patterns=pd.read_csv('learned_patterns.csv')
# update_learn_pattern('ATP', 21, (True,False,True))

# a=['Hello'] * 2
# print(a)
# print(np.full(5,4))
#
# df1=pd.DataFrame()
# df1=df1.append({'page':1,'doc':'pdf1','MP':'A B C'},ignore_index=True)
# df1=df1.append({'page':2,'doc':'pdf1','MP':'D E'},ignore_index=True)
# df1=df1.append({'page':3,'doc':'pdf1','MP':'K L'},ignore_index=True)
#
# df2=pd.DataFrame()
# df2=df2.append({'page':1,'doc':'pdf1','ATP':'H'},ignore_index=True)
# df2=df2.append({'page':1,'doc':'pdf1','ATP':'O L'},ignore_index=True)
# df2=df2.append({'page':2,'doc':'pdf2','ATP':'Z Y X'},ignore_index=True)
# df2=df2.append({'page':4,'doc':'pdf2','ATP':'M N P'},ignore_index=True)
#
#
# def agg_for_doc(series):
#     return list(series)[0]
#
#
# def agg_for_entity_values(series):
#     return ' '.join(series)
#
# def agg_for_page(series):
#     return list(series)[0]
#
#
# df2 = df2.groupby('page').agg(
#     {'doc': agg_for_doc, 'ATP':  agg_for_entity_values, 'page': agg_for_page})
#
# # print(df2)
#
# result = pd.merge(df1, df2, how='outer', on=['page', 'doc'])
#
# # print(result)
#
# df3=pd.DataFrame()
# df3=df3.append({'page':1,'doc':'pdf2','MP':'H'},ignore_index=True)
# df3=df3.append({'page':2,'doc':'pdf2','MP':'Z Y X'},ignore_index=True)
# df2=df2.append({'page':4,'doc':'pdf2','MP':'M N P'},ignore_index=True)
#
# result = pd.merge(result, df3, how='outer',on=['page', 'doc'])
#
#
# new_names={}
# for col_name in list(result.columns):
#     if '_x' in col_name or '_y' in col_name:
#         new_names[col_name]=col_name[:-2]
# result=result.rename(index=str, columns=new_names)
# def sjoin(x): return ';'.join(x[x.notnull()].astype(str))
# result=result.groupby(level=0, axis=1).apply(lambda x: x.apply(sjoin, axis=1))
# print(result)

# df4=pd.DataFrame()
# df4=df4.append({'page':1,'doc':'pdf2','ATP':'K','MP':''},ignore_index=True)
# df4=df4.append({'page':1,'doc':'pdf2','ATP':'','MP':'Z Y X'},ignore_index=True)
# df4=df4.append({'page':4,'doc':'pdf3','ATP':'D','MP':'M N P'},ignore_index=True)
#
# agg_dict={}
# agg_func=lambda x: ' '.join(x)
# for column in df4:
#     if column!='page' and column!='doc':
#         agg_dict[column]=agg_func
#
# print(df4.groupby(['page','doc']).agg(agg_dict))


# learned_patterns=pd.read_csv('learned_patterns.csv',index_col=0)
# learned_patterns = learned_patterns.replace(np.nan, '', regex=True)
# all_patterns=pd.read_csv('all_patterns.csv',index_col=0)
# all_patterns = all_patterns.replace(np.nan, '', regex=True)
# build_report()

# parsed = parse_document("/Users/akdidier/Documents/Drilling-Grant/pdfs/Anadarko/AVALANCHE_29-40_UNIT_1H_Completion_Reports.pdf")
# create_patterns_per_doc(parsed)
line = ["AVALANCHE 29-40 UNIT 1H/RICOCHET 29-40 UNIT 1H"]
create_patterns_per_doc(line)
