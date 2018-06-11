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

    # could be made more robust by taking the list of units from grobid. This is certainly not comprehensive
    units = ['ft', 'gal', 'ppa', 'psi', 'lbs', 'lb', 'bpm', 'bbls', 'bbl', '\'', "\"", "'", "°", "$", 'hrs']
    learned_patterns, all_patterns, current_patterns, interesting_patterns, fuzzy_patterns = [], [], [], [], []

    def __init__(self):
        self.current_patterns = pd.DataFrame(columns=['pattern_id', 'base_pattern', 'instances', 'hpattern',
                                                      'document_name', 'num_instances', 'mask','page_numbers'])
        self.threshold = 1

    def getID(self):
        """
        creates an id for each pattern by incrementing the class var counter
        :return: counter
        """
        KnowledgeExtractor.counter += 1
        return KnowledgeExtractor.counter

    def break_natural_boundaries(self, text):

        stringbreak = []
        if len(text.split(' ')) > 1:
            stringbreak = text.split(' ')
        else:
            # spl = '[\.\,|\%|\$|\^|\*|\@|\!|\_|\-|\(|\)|\:|\;|\'|\"|\{|\}|\[|\]|]'
            alpha = '[A-z]'
            num = '\d'
            spl = '[^A-z\d]'

            matchindex = set()
            matchindex.update(set(m.start() for m in re.finditer(num + alpha, text)))
            matchindex.update(set(m.start() for m in re.finditer(alpha + num, text)))
            matchindex.update(set(m.start() for m in re.finditer(spl + alpha, text)))
            matchindex.update(set(m.start() for m in re.finditer(alpha + spl, text)))
            matchindex.update(set(m.start() for m in re.finditer(spl + num, text)))
            matchindex.update(set(m.start() for m in re.finditer(num + spl, text)))
            matchindex.update(set(m.start() for m in re.finditer(spl + spl, text)))

            matchindex.add(len(text) - 1)
            matchindex = sorted(matchindex)
            start = 0

            for i in matchindex:
                end = i
                stringbreak.append(text[start:end + 1])
                start = i + 1
        stringbreak = [s for s in stringbreak if s]
        return stringbreak

    def break_and_split(self, arr):
        new_arr = []
        for token in arr:
            new_arr.extend(self.break_natural_boundaries(token))
        return new_arr

    def ispunc(self, s):
        if re.match('[^a-zA-Z\d]', s):
            return True
        return False

    def get_base_pattern(self, hpattern):
        '''
        takes the second level of an hpattern (non variable tokens)
        :param hpattern:
        :return:
        '''
        base_pattern = []
        for patt in hpattern:
            base_pattern.append(patt[1])

        return tuple(base_pattern)

    def has_numeric(self, token):
        if re.search(r"\d", token):
            return True
        else:
            return False



    def create_hpattern(self, instance):
        '''
        creates a heirarchy of 'denominations/classes' for each base pattern
        :param instance: a string? TODO double check with Asitang's original code
        :return: base_pattern, h_pattern
        '''

        signature = []

        for token in instance:
            #print(token)
            if token in KnowledgeExtractor.prepos:
                signature.append((token, token, KnowledgeExtractor.PREP))
            # elif token.isnumeric():
            elif self.has_numeric(token):
                signature.append((token, KnowledgeExtractor.DIGI, KnowledgeExtractor.DIGI))
            elif token.isalpha():
                sign = [token, token, KnowledgeExtractor.WORD]
                if token.lower() in KnowledgeExtractor.units:
                    sign.append(KnowledgeExtractor.UNIT)
                signature.append(tuple(sign))

            #maybe use spacy or nltk instead of ispunc
            elif self.ispunc(token):
                signature.append((token, token, KnowledgeExtractor.PUNC))
            else:
                if token:
                    signature.append(tuple(token))

        return tuple(signature)

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

    def get_pruned_indices(self, patterns, counted_patterns):
        """
        In all of the found patterns, we want to prune away some patterns that are subsets of each other, keeping the
        most detailed pattern if the support for the pattern and its subpattern are the same. For instance, suppose
        'max press: ## unit' is seen 3 times and 'max press: ##' is seen 3 times, then we want to keep the longer pattern
        because it has more information. If instead the shorter pattern were seen more times, the longer pattern is then
        a special case and we should keep both to prevent throwing away valuable information.
        :param patterns, list
        :param counted_patterns, collections.counter object
        :return:
        """
        pattern_indices = {patt: idx for idx, patt in enumerate(patterns)}
        counted_copy = copy.deepcopy(counted_patterns)
        for pattern in counted_patterns.keys():
            # create all n-gram subpatterns
            subpatterns = list(itertools.chain(*[list(ngrams(pattern, i)) for i in range(1, len(pattern))]))
            for subpat in subpatterns:
                if subpat in counted_copy.keys() and counted_copy[subpat] == counted_copy[pattern]:
                    counted_copy.pop(subpat)

        # print("final patterns ", counted_copy)
        final_patterns = list(counted_copy.keys())
        #print(final_patterns)
        final_patterns_indices = [pattern_indices[pat] for pat in final_patterns]
        return final_patterns_indices

    def significance_filter(self, row):
        '''
        Determines if a pattern is below the defined frequency threshold. Used to remove insignificant patterns.
        :param row, a row of the self.current_patterns
        :return bool
        '''
        global threshold
        # if the pattern occurs in the document less than the threshold then return false
        if int(row['num_instances']) > self.threshold:
            return True
        return False

    def punc_filter(self, row):
        '''
        Checks that the pattern does NOT start with punctuation or a preposition
        :param row, a row of the self.current_patterns
        :return bool
        '''
        print(row)
        pattern = ast.literal_eval(str(row['hpattern']))
        print(pattern)
        # if the first token is preposition/pronoun or punctuation then return false
        if pattern[0][2] == KnowledgeExtractor.PREP or pattern[0][2] == KnowledgeExtractor.PUNC:
            return False

        return True

    def no_entity_filter(self, row):
        '''
        Checks that the pattern contains an entity (Filter out patterns that contain only
        punctuations, prepositions, and numbers)
        row, a row of the self.current_patterns
        :return bool
        '''

        pattern = ast.literal_eval(str(row['hpattern']))
        for token in pattern:
            # if atleast one entity/unit found, it is okay
            if token[2] == KnowledgeExtractor.WORD:
                return True
        return False

    def no_num_filter(self, row):
        '''
        Checks that the pattern contains numbers
        row, a row of the self.current_patterns
        won't pick up any patterns that are strings
        :return bool
        '''

        pattern = ast.literal_eval(str(row['hpattern']))
        for token in pattern:
            # if at least one number found, it is okay
            if token[2] == KnowledgeExtractor.DIGI:
                return True
        return False

    def apply_filters(self):
        '''
        Apply filters to remove 'irrelevant' current patterns: see filter1 impl
        :param: filter: a function
        :return:
        '''

        filters = [self.significance_filter, self.punc_filter, self.no_entity_filter, self.no_num_filter]

        for f in filters:
            # print(f)
            # for index, row in self.current_patterns.iterrows():
            #     print("row ", row)
            #     try:
            #         print("output of filter", f(row))
            #     except Exception as e:
            #         print(e)
            self.current_patterns = self.current_patterns[self.current_patterns.apply(lambda x: f(x), axis=1)]
            if self.current_patterns.empty:
                print("Uh oh, we've filtered out everything. There were no meaningful patterns. "
                      "Try re-running with a lower significance threshold.")
                break

        print('FILTERED! now number of patterns: ', len(self.current_patterns))

    def create_patterns_per_doc(self, parsed_text, doc_name):
        '''

        :param parsed_text: it should be a list of texts. One item/text for every page in the document.
        :return:
        '''
        #check for expected input
        if isinstance(parsed_text, str):
            parsed_text = [parsed_text]
        elif not isinstance(parsed_text, list):
            raise ValueError("Expected type list. Got type ", type(parsed_text))

        global current_patterns
        global current_document_path
        global all_patterns

        all_hpatterns = []
        all_base_patterns = []
        all_instances = []

        for page in parsed_text:
            page_hpatterns = []
            page_base_patterns = []
            page_instances = []
            for line in page.split('\n'):  # pattern analysis is done based on each line
                if not line:
                    continue
                # create chunks by dividing on commas+space, period+space (and multi-space??) so that patterns don't span beyond them
                chunks = re.split(', |\. |;', line.lower())
                # remove commas from numbers (8,643), give valid spacing around #, = and @
                # tokenize everything based on spaces/tabs
                # creates a list(token) of lists(chunk): [[token,token,token],[token,token]]

                chunks = [re.sub(r"([^a-zA-Z0-9])", r" \1 ", chunk.replace(",", "")) for chunk in chunks]

                # separate the tokens further using the natural separation boundaries
                chunks = [self.break_natural_boundaries(chunk) for chunk in chunks]
               # print("chunks ", chunks)
                chunks_base_patterns = []
                chunks_hpatterns = []

                for chunk in chunks:
                    if not chunk:
                        continue
                 #   print(chunk)
                    # convert each chunk to base pattern and hpattern
                    hpattern = self.create_hpattern(chunk)
                    # print("hpattern", hpattern)
                    if not hpattern:
                        print("skipping empty pattern")
                        continue
                    base_pattern = self.get_base_pattern(hpattern)
                    # print("base pattern", base_pattern)
                    chunks_base_patterns.append(base_pattern)
                    chunks_hpatterns.append(hpattern)

                # create n-grams

                n_gram_range = (3, 4, 5, 6, 7)
                for n in n_gram_range:
                    all_grams_base_patterns = list(map(lambda x: list(ngrams(x, n)), chunks_base_patterns))
                    all_grams_hpatterns = list(map(lambda x: list(ngrams(x, n)), chunks_hpatterns))
                    all_grams = list(map(lambda x: list(ngrams(x, n)), chunks))

                    # flatten the nested list
                    all_grams_base_patterns = [item for sublist in all_grams_base_patterns for item in sublist]
                    all_grams_hpatterns = [item for sublist in all_grams_hpatterns for item in sublist]
                    all_grams = [item for sublist in all_grams for item in sublist]

                    page_base_patterns.extend(all_grams_base_patterns)
                    page_hpatterns.extend(all_grams_hpatterns)
                    page_instances.extend(all_grams)

            all_base_patterns.append(page_base_patterns)
            all_hpatterns.append(page_hpatterns)
            all_instances.append(page_instances)

        all_page_numbers = []
        for indx, _ in enumerate(all_instances):
            all_page_numbers.append(list(np.full(len(_), indx + 1)))

        all_base_patterns_flattened = [item for sublist in all_base_patterns for item in sublist]
        all_hpatterns_flattened = [item for sublist in all_hpatterns for item in sublist]
        all_instances_flattened = [item for sublist in all_instances for item in sublist]
        all_page_numbers_flattened = [item for sublist in all_page_numbers for item in sublist]

        all_pats_list = [all_base_patterns_flattened, all_hpatterns_flattened, all_instances_flattened, all_page_numbers_flattened]
        # ======= get the longest pattern with the same support (keeps only the superset, based on minsup criteria)
        counted_patterns = collections.Counter(all_base_patterns_flattened)
        pruned_indices = self.get_pruned_indices(all_base_patterns_flattened, counted_patterns)
        all_pruned = list(map(lambda x: [x[i] for i in pruned_indices], all_pats_list))

        # ========== create data frame

        # aggregate the instances based on base patterns
        # create a mapping from base pattern to hpattern
        aggregated_pattern_instance_mapping = {}
        aggregated_pattern_pagenumber_mapping = {}
        base_pattern_to_hpattern = {}

        for pattern, hpattern, instance, page_number in zip(all_pruned[0], all_pruned[1], all_pruned[2], all_pruned[3]):
            # aggregate
            if pattern not in aggregated_pattern_instance_mapping.keys():
                aggregated_pattern_instance_mapping[pattern] = []
                aggregated_pattern_pagenumber_mapping[pattern] = []

            aggregated_pattern_instance_mapping[pattern].append(instance)
            aggregated_pattern_pagenumber_mapping[pattern].append(page_number)

            # mapping
            if pattern not in base_pattern_to_hpattern.keys():
                base_pattern_to_hpattern[pattern] = hpattern

        for pattern in aggregated_pattern_instance_mapping.keys():
            #print("adding to df")
            self.current_patterns = self.current_patterns.append({'pattern_id': self.getID(),
                                                    'base_pattern': str(pattern),
                                                    'instances': str(aggregated_pattern_instance_mapping[pattern]),
                                                    'page_numbers': aggregated_pattern_pagenumber_mapping[pattern],
                                                    'hpattern': str(base_pattern_to_hpattern[pattern]),
                                                    'document_name': doc_name,
                                                    'num_instances': counted_patterns[pattern]},
                                                   ignore_index=True)

        #filter garbage patterns
        self.apply_filters()

        #save the patterns
        # self.current_patterns = self.current_patterns.replace(np.nan, '', regex=True)
        # sel.fcurrent_patterns.to_csv('current_patterns.csv')
