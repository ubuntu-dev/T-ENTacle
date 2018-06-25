import string
import ast
import numpy as np
import re
from fuzzywuzzy import fuzz
from nltk import ngrams
import collections
import copy
import itertools
import pandas as pd
from pattern import Pattern
import spacy
import dill
import timeit
import os

"""
right now, works on one document at a time, but could move patterns dataframe to a class variable 
so that all instances can modify it. But it seems like a better idea to concatenate the output from several
instances, or to set up some sort of inheritance structure. 

Check for entities like currency, date, numbers
See if there is a better way to tokenize with spacy or something

TODO: 
1. consider making threshold mutable by the user
2. currently filters out patterns that don't have numbers. Are there any strings that should be left in?
"""
class KnowledgeExtractor(object):
    PREP = "Prep~"
    PUNC = 'Punc~'
    WORD = 'Word~'
    DIGI = 'Digi~'
    UNIT = 'Unit~'
    prepos = ['aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'anti', 'around',
                   'as',
                   'at', 'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by',
                   'concerning', 'considering', 'despite', 'down', 'during', 'except', 'excepting', 'excluding',
                   'following',
                   'for', 'from', 'in', 'inside', 'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto',
                   'opposite',
                   'outside',
                   'over', 'past', 'per', 'plus', 'regarding', 'round', 'save', 'since', 'than', 'through', 'to',
                   'toward',
                   'towards',
                   'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via', 'with', 'within', 'without',
                   'and', 'or']
    punc = set(string.punctuation)
    counter = 0
    nlp = spacy.load('en')
    # could be made more robust by taking the list of units from grobid. This is certainly not comprehensive
    units = ['ft', 'gal', 'ppa', 'psi', 'lbs', 'lb', 'bpm', 'bbls', 'bbl', '\'', "\"", "'", "°", "$", 'hrs']
    learned_patterns, all_patterns, current_patterns, interesting_patterns, fuzzy_patterns = [], [], [], [], []

    def __init__(self, threshold=10):
        # save state across documents
        if os.path.exists('counter'):
            with open('counter', 'rb') as f:
                self.counter = dill.load(f)
        else:
            self.counter = 0
        ##################################
        #these should perhaps become class variables
        # entity matchings for all the documents processed so far
        if os.path.exists('learned_patterns.csv'):
            self.learned_patterns = pd.read_csv('learned_patterns.csv', index_col=0)
            self.learned_patterns.replace(np.nan, '', regex=True, inplace=True)
        else:
            self.learned_patterns = pd.DataFrame(columns=['entity_name', 'seed_aliases', 'pattern_ids'])

        # pattern information about all the patterns seen so far from all the documents processed
        if os.path.exists('all_patterns.csv'):
            self.all_patterns = pd.read_csv('all_patterns.csv', index_col=0)
            self.all_patterns.replace(np.nan, '', regex=True, inplace=True)
        else:
            self.all_patterns = pd.DataFrame(
                columns=['pattern_id', 'base_pattern', 'instances', 'hpattern', 'document_name', 'num_instances',
                         'mask', 'page_numbers'])
        #####################################

        self.current_patterns = pd.DataFrame(columns=['pattern_id', 'base_pattern', 'instances', 'hpattern',
                                                      'document_name', 'num_instances', 'mask','page_numbers'])
        self.threshold = threshold

    def getID(self):
        """
        creates an id for each pattern by incrementing the class var counter
        :return: counter
        """
        KnowledgeExtractor.counter += 1
        return KnowledgeExtractor.counter


    def clean_text(self, text):
        text = re.sub(r"(\$)(\s)(\d)", r"\1 \3", text)
        return text


    def break_natural_boundaries(self, text):
        # text = self.whitespace_around_punc(text)
        # tokenized = text.split()
        # for t in tokenized:
        #     print(t)
        #     doc = self.nlp(t)
        #     for ent in doc.ents:
        #         print(ent.text, ent.label_)
        parsed = self.nlp(text)
        tokenized = [str(token) for token in parsed]
        return tokenized
        #
        # print("\nEntities")
        # for ent in parsed.ents:
        #     print(ent.text, ent.label_)
        # print(text.split())

    # def break_natural_boundaries(self, text):
    #
    #     stringbreak = []
    #     if len(text.split(' ')) > 1:
    #         stringbreak = text.split(' ')
    #     else:
    #         # spl = '[\.\,|\%|\$|\^|\*|\@|\!|\_|\-|\(|\)|\:|\;|\'|\"|\{|\}|\[|\]|]'
    #         alpha = '[A-z]'
    #         num = '\d'
    #         spl = '[^A-z\d]'
    #
    #         matchindex = set()
    #         matchindex.update(set(m.start() for m in re.finditer(num + alpha, text)))
    #         matchindex.update(set(m.start() for m in re.finditer(alpha + num, text)))
    #         matchindex.update(set(m.start() for m in re.finditer(spl + alpha, text)))
    #         matchindex.update(set(m.start() for m in re.finditer(alpha + spl, text)))
    #         matchindex.update(set(m.start() for m in re.finditer(spl + num, text)))
    #         matchindex.update(set(m.start() for m in re.finditer(num + spl, text)))
    #         matchindex.update(set(m.start() for m in re.finditer(spl + spl, text)))
    #
    #         matchindex.add(len(text) - 1)
    #         matchindex = sorted(matchindex)
    #         start = 0
    #
    #         for i in matchindex:
    #             end = i
    #             stringbreak.append(text[start:end + 1])
    #             start = i + 1
    #     stringbreak = [s for s in stringbreak if s]
    #     for s in stringbreak:
    #         print(s)
    #     return stringbreak


    def find_entities(self, hpattern, mask=[]):
        '''
        aggrerate the words that are next to each other as an entity. Find multiple ones.
        :param hpattern:
        :return:
        '''

        if len(mask) == 0:
            mask = list(np.full(len(hpattern), True))

        entities = []
        entity = ''
        dummied_hpatteren = list(hpattern)
        dummied_hpatteren.append(('~', '~', '~'))
        dummied_hpatteren = tuple(dummied_hpatteren)
        mask.append(True)

        for token, select in zip(dummied_hpatteren, mask):
            if not select:
                continue
            if token[2] == self.WORD:
                entity += ' ' + token[0]
            else:
                if entity != '':
                    entities.append(entity)
                entity = ''
        return entities

    def find_units(self, hpattern, mask=[]):
        '''
        find the units in the pattern
        :param hpattern:
        :return:
        '''

        if len(mask) == 0:
            mask = list(np.full(len(hpattern), True))

        units = []
        for token, select in zip(hpattern, mask):
            if not select:
                continue
            if len(token) >= 4 and token[3] == self.UNIT:
                units.append(token[0])
        return units

    def find_values(self, instance, hpattern, mask=[]):
        '''
        find the values in the pattern
        :param hpattern:
        :return:
        '''

        values = []
        if len(mask) == 0:
            mask = list(np.full(len(hpattern), True))

        for token_inst, token_patt, select in zip(instance, hpattern, mask):
            if not select:
                continue
            if token_patt[2] == self.DIGI:
                values.append(token_inst)
        return values

    def find_exact_patterns(self, hpattern):
        '''
        finds the hpatterns that are exact to the given hpattern
        look by base patterns, as they don't have the variable/value
        :param hpattern:
        :return:
        '''
        global current_patterns
        exact_pattern_ids = []
        base_pattern = str(self.get_base_pattern(ast.literal_eval(hpattern)))
        if base_pattern in list(current_patterns['base_pattern']):
            exact_pattern_ids.append(
                list(current_patterns[current_patterns['base_pattern'] == base_pattern]['pattern_id'])[0])
        return exact_pattern_ids

    def find_close_patterns(self, hpattern):
        '''
        finds the hpatterns that are closest to the given hpattern
        :param hpattern:
        :return:
        '''
        global current_patterns
        close_pattern_ids = []

        #WHY IS ASITANG USING THIS?
        hpattern = ast.literal_eval(hpattern)
        entities = self.find_entities(hpattern)
        units = self.find_units(hpattern)

        close_patterns = []

        for _, row in current_patterns.iterrows():
            confidence_flag_entity = 0
            confidence_flag_unit = 0
            confidence = 0  # todo: give the best score here; will help decide the rank
            hpattern_iter = ast.literal_eval(str(row['hpattern']))
            mask = str(row['mask'])
            if mask == '':
                mask = []
            else:
                mask = ast.literal_eval(str(row['mask']))
            entities_iter = self.find_entities(hpattern_iter, mask)
            units_iter = self.find_units(hpattern_iter, mask)

            for entity_iter in entities_iter:
                for entity in entities:
                    if fuzz.ratio(entity, entity_iter) > 70:
                        confidence_flag_entity = 1

            for unit_iter in units_iter:
                for unit in units:
                    if unit.lower() == unit_iter.lower():
                        confidence_flag_unit = 1

            if confidence_flag_entity == 1 or confidence_flag_unit == 1:
                close_patterns.append((row['pattern_id'], confidence_flag_entity, confidence_flag_unit))

        # todo: here rank the patterns according to confidence and return the top n

        for conf in close_patterns:
            close_pattern_ids.append(conf[0])

        return close_pattern_ids


    def prune(self, patterns):
        """
        In all of the found patterns, we want to prune away some patterns that are subsets of each other, keeping the
        most detailed pattern if the support for the pattern and its subpattern are the same. For instance, suppose
        'max press: ## unit' is seen 3 times and 'max press: ##' is seen 3 times, then we want to keep the longer pattern
        because it has more information. If instead the shorter pattern were seen more times, the longer pattern is
        likely a special case and we should keep both to prevent throwing away valuable information.
        :param patterns, list of pattern objects
        :return:
        """
        counted_patterns = collections.Counter([p.base_pattern for p in patterns])
        pattern_indices = {patt: idx for idx, patt in enumerate([p.base_pattern for p in patterns])}
        pattern_indices = {}
        for idx, patt in enumerate([p.base_pattern for p in patterns]):
            if patt not in pattern_indices:
                pattern_indices[patt] = [idx]
            else:
                pattern_indices[patt].append(idx)

        to_remove = []
        for p in counted_patterns.keys():
            if p in to_remove:
                continue
            subpatterns = list(itertools.chain(*[list(ngrams(p, i)) for i in range(3, len(p))]))
            #If a subpattern has the same support as the pattern, remove the subpattern.
            for subpat in subpatterns:
                if subpat == p:
                    continue
                #print("Checking ", subpat)
                if subpat in counted_patterns.keys() and counted_patterns[subpat] == counted_patterns[p]:
                    #print("pattern and subpat have same support")
                    #print("Removing ", subpat)
                    if subpat not in to_remove:
                        to_remove.append(subpat)

        # print("Removing patterns:")
        # print(to_remove)
        for p in to_remove:
            # print(p)
            counted_patterns.pop(p)
        final_patterns = list(counted_patterns.keys())
        final_indices = []
        for pat in final_patterns: final_indices += pattern_indices[pat]
        final_pattern_objects = [patterns[i] for i in final_indices]
        return final_pattern_objects

    def aggregate_patterns(self, patterns):
        """
        Looks for patterns with the same base pattern and stores their instances together
        :param patterns: list of pattern objects
        :return:
        """
        aggregated = {}
        for p in patterns:
            if p.base_pattern not in aggregated.keys():
                aggregated[p.base_pattern] = p
            else:
                aggregated[p.base_pattern].add_instance(p.instance)
                aggregated[p.base_pattern].add_page_num(p.page_num)
        return list(aggregated.values())

    def significance_filter(self, row):
        '''
        Determines if a pattern is below the defined frequency threshold. Used to remove insignificant patterns.
        :param row, a row of the self.current_patterns
        :return bool
        '''
        # TODO rewrite this
        # if the pattern occurs in the document less than the threshold then return false
        if int(row['num_instances']) > self.threshold:
            return True
        return False

    def punc_filter(self, pattern):
        '''
        Checks that the pattern does NOT start with punctuation or a preposition
        :param pattern, a pattern object
        :return bool
        '''
        # if the first token is preposition/pronoun or punctuation then get rid of it
        if pattern.hpattern[0][2] == Pattern.PREP or pattern.hpattern[0][2] == Pattern.PUNC:
            return False
        return True

    def no_entity_filter(self, pattern):
        '''
        Checks that the pattern contains an entity (Filter out patterns that contain only
        punctuations, prepositions, and numbers)
        :param pattern, a pattern object
        :return bool
        '''

        labels = [h[-1] for h in pattern.hpattern]
        if Pattern.WORD in labels:
            return True
        return False

    def multiple_entities_filter(self, pattern):
        """
        KNOWN BUG: removes dates in format MM/DD/YYYY or nums in format ###.## bc knowledge facilitator breaks things
        on periods
        if a pattern contains more than one number or unit, then multiple entities have been concatenated into one chunk
        :param pattern: a pattern object
        :return:
        """
        labels = [h[-1] for h in pattern.hpattern]
        label_counts = collections.Counter(labels)
        if label_counts[pattern.DIGI] > 1 or label_counts[pattern.UNIT] > 1:
            #get rid of it
            return False
        return True

    def no_num_filter(self, pattern):
        '''
        Checks that the pattern contains numbers
        row, a row of the self.current_patterns
        won't pick up any patterns that are strings
        :param pattern, a pattern object
        :return bool
        '''

        # if at least one number found, it is okay
        labels = [h[-1] for h in pattern.hpattern]
        if Pattern.DIGI in labels:
            return True
        return False

    def apply_filters(self, filters, patterns):
        '''
        Apply filters to remove 'irrelevant' patterns
        :param: filters, list of filter functions
        :param: patterns, list of pattern objects
        :return:
        '''
        print(len(patterns))
        for f in filters:
            patterns = list(filter(f, patterns))
            print(len(patterns))
            if not patterns:
                print("Uh oh, we've filtered out everything. There were no meaningful patterns. ")
        print('FILTERED! now number of patterns: ', len(patterns))
        return patterns

    def create_patterns_per_doc(self, parsed_text, doc_name = None):
        '''
        The main driver of knowledge extractor which searches for patterns in a given text.
        :param parsed_text: it should be a list of texts. One item/text for every page in the document.
        :param doc_name: string, optional. The name of the document being parsed.
        :return: None
        '''
        #check for expected input
        if isinstance(parsed_text, str):
            parsed_text = [parsed_text]
        elif not isinstance(parsed_text, list):
            raise ValueError("Expected type list. Got type ", type(parsed_text))

        patterns = [] #To store all the pattern objects for the doc

        #Create the signatures for each pattern. All signature information is stored in a pattern object.
        for page_num, page in enumerate(parsed_text):

            page = self.clean_text(page)
            for line in page.split('\n'):  # pattern analysis is done based on each line
                if not line:
                    continue
                # create chunks by dividing on commas+space, period+space (and multi-space??)
                # so that patterns don't span beyond them
                chunks = re.split(',\s{1,5}|\.\s{1,5}|;', line.lower())
                #uncomment this line to use the non-spacy version of tokenizing
                #chunks = [re.sub(r"([^a-zA-Z0-9])", r" \1 ", chunk.replace(",", "")) for chunk in chunks]

                # separate the tokens further using the natural separation boundaries (spacy version)
                chunks = [self.break_natural_boundaries(chunk) for chunk in chunks]

                #create n-grams out of every chunk and patterns for each n-gram
                n_gram_range = (3, 4, 5, 6, 7)
                for chunk in chunks:
                    if not chunk:
                        continue
                    all_grams_nested = list(map(lambda n: list(ngrams(chunk, n)), n_gram_range))
                    #flatten because ngrams returns a list of tuples
                    all_grams = []
                    for lst in all_grams_nested: all_grams += lst
                    #print("all grams ", all_grams)
                    n_gram_patterns = list(map(lambda text: Pattern(text, page_num, doc_name), all_grams)) #list of pattern objects
                    patterns.extend(n_gram_patterns)

        #Filter, prune, and aggregate pattern instances
        filtered_patterns = self.apply_filters([self.punc_filter, self.no_num_filter, self.no_entity_filter,
                                                self.multiple_entities_filter], patterns)
        #get the longest pattern with the same support (keeps only the superset, based on minsup criteria)
        pattern_subset = self.prune(filtered_patterns)

        # #aggregate all of the patterns
        aggregated_patterns = self.aggregate_patterns(pattern_subset)

        #create a dataframe of all the patterns
        self.current_patterns = pd.concat([self.current_patterns,
                                           pd.DataFrame(list(map(lambda p: p.get_dict(), aggregated_patterns)))])

        #save the patterns
        self.current_patterns = self.current_patterns.replace(np.nan, '', regex=True)
        self.current_patterns.to_csv('current_patterns.csv', index=False)
