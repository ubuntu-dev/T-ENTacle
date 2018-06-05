import string
import ast
import numpy as np
import re
from fuzzywuzzy import fuzz

class KnowledgeExtractor:
    def __init___(self):
        self.PREP = "Prep~"
        self.PUNC = 'Punc~'
        self.WORD = 'Word~'
        self.DIGI = 'Digi~'
        self.UNIT = 'Unit~'
        self.prepos = ['aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'anti', 'around',
              'as',
              'at', 'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by',
              'concerning', 'considering', 'despite', 'down', 'during', 'except', 'excepting', 'excluding', 'following',
              'for', 'from', 'in', 'inside', 'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto', 'opposite',
              'outside',
              'over', 'past', 'per', 'plus', 'regarding', 'round', 'save', 'since', 'than', 'through', 'to', 'toward',
              'towards',
              'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via', 'with', 'within', 'without',
              'and', 'or']
        self.punc = set(string.punctuation)

        #could be made more robust by taking the list of units from grobid. This is certainly not comprehensive
        self.units = ['ft', 'gal', 'ppa', 'psi', 'lbs', 'lb', 'bpm', 'bbls', 'bbl', '\'', "\"", "'", "°", "$", 'hrs']
        self.learned_patterns, self.all_patterns, self.current_patterns, self.interesting_patterns, self.fuzzy_patterns\
            = [], [], [], [], []



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
        return stringbreak

    def break_and_split(self, arr):
        new_arr = []
        for token in arr:
            new_arr.extend(self.break_natural_boundaries(token))
        return new_arr

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

    def create_patterns_per_doc(self, parsed_text):
        '''

        :param parsed_text: it should be a list of texts. One item/text for every page in the document.
        :return:
        '''
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
                chunks = [self.break_and_split(chunk) for chunk in chunks]
                print(chunks)
                chunks_base_patterns = []
                chunks_hpatterns = []

                for chunk in chunks:
                    # convert each chunk to base pattern and hpattern
                    hpattern = self.create_hpattern(chunk)
                    print("hpattern", hpattern)
                    base_pattern = self.get_base_pattern(hpattern)
                    print("base pattern", base_pattern)
                    chunks_base_patterns.append(base_pattern)
                    chunks_hpatterns.append(hpattern)

                # create n-grams

        #         n_gram_range = (3, 4, 5, 6, 7)
        #         for n in n_gram_range:
        #             all_grams_base_patterns = list(map(lambda x: list(ngrams(x, n)), chunks_base_patterns))
        #             all_grams_hpatterns = list(map(lambda x: list(ngrams(x, n)), chunks_hpatterns))
        #             all_grams = list(map(lambda x: list(ngrams(x, n)), chunks))
        #
        #             # flatten the nested list
        #             all_grams_base_patterns = [item for sublist in all_grams_base_patterns for item in sublist]
        #             all_grams_hpatterns = [item for sublist in all_grams_hpatterns for item in sublist]
        #             all_grams = [item for sublist in all_grams for item in sublist]
        #
        #             page_base_patterns.extend(all_grams_base_patterns)
        #             page_hpatterns.extend(all_grams_hpatterns)
        #             page_instances.extend(all_grams)
        #
        #     all_base_patterns.append(page_base_patterns)
        #     all_hpatterns.append(page_hpatterns)
        #     all_instances.append(page_instances)
        #
        # all_page_numbers = []
        # for indx, _ in enumerate(all_instances):
        #     all_page_numbers.append(list(np.full(len(_), indx + 1)))
        #
        # all_base_patterns_flattened = [item for sublist in all_base_patterns for item in sublist]
        # all_hpatterns_flattened = [item for sublist in all_hpatterns for item in sublist]
        # all_instances_flattened = [item for sublist in all_instances for item in sublist]
        # all_page_numbers_flattened = [item for sublist in all_page_numbers for item in sublist]
        #
        # counted_patterns = collections.Counter(all_base_patterns_flattened)
        #
        # # ======= get the longest pattern with the same support (keeps only the superset, based on minsup criteria)
        # # todo: check if works correctly
        # filtered_patterns = {}
        # for pattern in counted_patterns.keys():
        #     # create the ngrams/subsets of a set and check if they are already present, if so check minsup and delete
        #     len_pattern = len(pattern)
        #     filtered_patterns[pattern] = counted_patterns[pattern]
        #     for i in range(1, len_pattern):
        #         # create all size sub patterns/n-grams
        #         subpatterns = list(ngrams(pattern, i))
        #         for subpattern in subpatterns:
        #             if subpattern in filtered_patterns.keys() and filtered_patterns[subpattern] == counted_patterns[
        #                 pattern]:
        #                 # delete subpattern
        #                 # print('deleting',subpattern,', because: ', pattern, filtered_pattens[subpattern], counted[pattern])
        #                 filtered_patterns.pop(subpattern)
        #
        # # ========== create data frame
        #
        # # aggregate the instances based on base patterns
        # # create a mapping from base pattern to hpattern
        # aggregated_pattern_instance_mapping = {}
        # aggregated_pattern_pagenumber_mapping = {}
        # base_pattern_to_hpattern = {}
        # for pattern, hpattern, instance, page_number in zip(all_base_patterns_flattened, all_hpatterns_flattened,
        #                                                     all_instances_flattened, all_page_numbers_flattened):
        #
        #     # aggregate
        #     if pattern not in aggregated_pattern_instance_mapping.keys():
        #         aggregated_pattern_instance_mapping[pattern] = []
        #         aggregated_pattern_pagenumber_mapping[pattern] = []
        #
        #     aggregated_pattern_instance_mapping[pattern].append(instance)
        #     aggregated_pattern_pagenumber_mapping[pattern].append(page_number)
        #
        #     # mapping
        #     if pattern not in base_pattern_to_hpattern.keys():
        #         base_pattern_to_hpattern[pattern] = hpattern
        #
        # for pattern in aggregated_pattern_instance_mapping.keys():
        #     if pattern in filtered_patterns:
        #         pattern_id = getID()
        #
        #         current_patterns = current_patterns.append({'pattern_id': pattern_id, 'base_pattern': str(pattern),
        #                                                     'instances': str(
        #                                                         aggregated_pattern_instance_mapping[pattern]),
        #                                                     'page_numbers': aggregated_pattern_pagenumber_mapping[
        #                                                         pattern],
        #                                                     'hpattern': str(base_pattern_to_hpattern[pattern]),
        #                                                     'document_name': current_document,
        #                                                     'num_instances': counted_patterns[pattern]},
        #                                                    ignore_index=True)
        #
        # # ============= apply filters
        #
        # # filter the patterns that have the number of instances below a certain threshold
        # apply_filters(filter1)
        # # remove the ones that start with a punctuation or preposition
        # apply_filters(filter2)
        # # remove the patterns that have only punctuations, prepositions and numbers
        # apply_filters(filter3)
        # # remove the ones that have no numbers
        # apply_filters(filter4)
        #
        # current_patterns = current_patterns.replace(np.nan, '', regex=True)
        # current_patterns.to_csv('current_patterns.csv')
