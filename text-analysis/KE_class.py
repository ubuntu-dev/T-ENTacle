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
import utils
import locale

"""
right now, works on one document at a time, but could move patterns dataframe to a class variable 
so that all instances can modify it. But it seems like a better idea to concatenate the output from several
instances, or to set up some sort of inheritance structure. 

Check for entities like currency, date, numbers
See if there is a better way to tokenize with spacy or something

TODO: 
keep dictionary of pattern objects, key is id
1. consider making threshold mutable by the user
2. currently filters out patterns that don't have numbers. Are there any strings that should be left in?
3. add a way for user to add seed aliases
4. Think about the flow of when patterns are added to allpatterns. This should probably occur after a user is finished processing
    the current patterns.
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
    nlp = spacy.load('en')
    # could be made more robust by taking the list of units from grobid. This is certainly not comprehensive
    units = ['ft', 'gal', 'ppa', 'psi', 'lbs', 'lb', 'bpm', 'bbls', 'bbl', '\'', "\"", "'", "°", "$", 'hrs']
    learned_patterns, all_patterns, current_patterns, interesting_patterns, fuzzy_patterns = [], [], [], [], []

    def __init__(self, threshold=10):
        # save state across documents
        if os.path.exists('counter'):
            self.counter = utils.load('counter')
        else:
            self.counter = 0
        ##################################
        #these should perhaps become class variables
        # entity matchings for all the documents processed so far
        if os.path.exists('learned_patterns.csv'):
            self.learned_patterns_df = pd.read_csv('learned_patterns.csv', index_col=0)
            self.learned_patterns_df.replace(np.nan, '', regex=True, inplace=True)
        else:
            self.learned_patterns_df = pd.DataFrame(columns=['entity_name', 'seed_aliases', 'pattern_ids'])
        if os.path.exists("learned_patterns.pkl"):
            self.learned_patterns = utils.load("learned_patterns.pkl")
        else:
            self.learned_patterns = {}
        if os.path.exists("all_patterns.pkl"):
            print("loaded all patterns")
            self.all_patterns = utils.load("all_patterns.pkl")
        else:
            self.all_patterns = {}

        # pattern information about all the patterns seen so far from all the documents processed
        # if os.path.exists('all_patterns.csv'):
        #     self.all_patterns = pd.read_csv('all_patterns.csv', index_col=0)
        #     self.all_patterns.replace(np.nan, '', regex=True, inplace=True)
        # else:
        #     self.all_patterns = pd.DataFrame(
        #         columns=['pattern_id', 'base_pattern', 'instances', 'hpattern', 'document_name', 'num_instances',
        #                  'mask', 'page_numbers'])
        if os.path.exists('all_patterns.pkl'):
            self.all_patterns = utils.load('all_patterns.pkl') #dictionary of pattern objects. key is hpattern
        else:
            self.all_patterns = {}
        #####################################
        #TODO refactor, move away from DF of current_patterns and move to list of pattern objects
        self.current_patterns = pd.DataFrame(columns=['base_pattern', 'instances', 'hpattern',
                                                      'document_name', 'num_instances', 'mask','page_numbers'])
        self.current_patterns.index.name = 'id'
        self.threshold = threshold

        self.curr_patterns = {}


    def makeID(self):
        """
        creates an id for each pattern by incrementing the class var counter
        :return: counter
        """
        self.counter += 1
        return self.counter


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
        tokenized = []
        for token in parsed:
            if re.search("\d", str(token)):
                token = str(token)
                symbols = re.compile(r"[^\d,\.a-zA-Z_\/]")
                if re.search(symbols, token):
                    token = re.sub(r"([^\d,\.a-zA-Z_\/])", " \1 ", token )
                    toks = token.split()
                    toks = [t for t in toks if t]
                    tokenized.extend(toks)
                else:
                    tokenized.append(token)
            elif not token.is_space:
                tokenized.append(str(token))
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
                aggregated[p.base_pattern].add(p.location)
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
        try:
            if pattern.hpattern[0][2] == Pattern.PREP or pattern.hpattern[0][2] == Pattern.PUNC:
                return False
        except IndexError:
            print("IndexError")
            print(pattern.instances)
            print(pattern.hpattern)
            exit(1)
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
        for f in filters:
            patterns = list(filter(f, patterns))
            if not patterns:
                print("Uh oh, we've filtered out everything. There were no meaningful patterns. ")
        print('FILTERED! now number of patterns: ', len(patterns))
        return patterns

    def add_curr_to_all_patterns(self):
        """
        Merges current_patterns with all_patterns. All previously found patterns get updated with the new instances
        found in the current document. This keeps the patterns aggregated by their base_pattern
        :return:
        """
        #check if all_patterns is empty. Then Knowledge Extractor has not learned anything
        #and we should just set all patterns to curr_patterns
        if not self.all_patterns:
            print("Setting allpatterns for the first time")
            self.all_patterns = copy.deepcopy(self.curr_patterns)
        else:
        #add the patterns in current patterns to all_patterns. The pattern objects in all_patterns
        #need to have their instances and document names updated.
            for basepattern in self.curr_patterns.keys():
                if basepattern in self.all_patterns:
                    #update all_patterns with the additional instances and documents
                    self.all_patterns[basepattern].add(self.curr_patterns[basepattern].location)
                else:
                    self.all_patterns[basepattern] = self.curr_patterns[basepattern]

    def create_patterns_per_doc(self, parsed_text, doc_name = None):
        '''
        The main driver of knowledge extractor which searches for patterns in a given text.
        :param parsed_text: it should be a list of texts. One item/text for every page in the document.
        :param doc_name: string, optional. The name of the document being parsed.
        :return: None
        '''
        print("Generating patterns.")
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
                # create chunks by dividing on commas+space,
                # EDIT: period+space (and multi-space??) was causing some patterns with abbreviations to be missed.
                #       This is removed in favor of line endings
                # so that patterns don't span beyond them
                chunks = re.split(',\s{1,5}|\n|;', line.lower())
                #uncomment this line to use the non-spacy version of tokenizing
                #chunks = [re.sub(r"([^a-zA-Z0-9])", r" \1 ", chunk.replace(",", "")) for chunk in chunks]

                # separate the tokens further using the natural separation boundaries (spacy version)
                chunks = [self.break_natural_boundaries(chunk) for chunk in chunks]

                #create n-grams out of every chunk and patterns for each n-gram
                n_gram_range = (2, 3, 4, 5, 6, 7)
                for chunk in chunks:
                    if not chunk:
                        continue
                    all_grams_nested = list(map(lambda n: list(ngrams(chunk, n)), n_gram_range))
                    #flatten because ngrams returns a list of tuples
                    all_grams = []
                    for lst in all_grams_nested: all_grams += lst
                    #print("all grams ", all_grams)
                    n_gram_patterns = list(map(lambda text: Pattern(text, page_num, doc_name, self.makeID()), all_grams)) #list of pattern objects
                    patterns.extend(n_gram_patterns)

        #Filter, prune, and aggregate pattern instances
        filtered_patterns = self.apply_filters([self.punc_filter, self.no_num_filter, self.no_entity_filter,
                                                self.multiple_entities_filter], patterns)
        #get the longest pattern with the same support (keeps only the superset, based on minsup criteria)
        pattern_subset = self.prune(filtered_patterns)

        #group all of the pattern instances by their base_pattern
        aggregated_patterns = self.aggregate_patterns(pattern_subset)

        #keep dictionary of pattern objects
        self.curr_patterns = {p.base_pattern: p for p in aggregated_patterns}

        #create a dataframe of all the patterns
        self.current_patterns = pd.concat([self.current_patterns,
                                           pd.DataFrame(list(map(lambda p: p.get_dict(), aggregated_patterns)))])

        #save the patterns
        self.current_patterns = self.current_patterns.replace(np.nan, '', regex=True)
        self.current_patterns.to_csv('current_patterns.csv')

        #merge patterns with all the other patterns. If we have seen the same patterns before, they should just
        #be added based on their base_pattern
        self.add_curr_to_all_patterns()
        self.save_all_patterns()
        #self.all_patterns = pd.concat([self.current_patterns, self.all_patterns])

    def save_all_patterns(self):
        """
        for testing
        :return:
        """
        with open("all_patterns.pkl", "wb") as f:
            dill.dump(self.all_patterns, f)

    def save(self, model):
        """
        Write learning/patterns to file
        :param model_name: an object model
        :return:
        """
        #TODO get the pwd and add to path. all_patterns doesn't seem to be saving
        base_path = os.getcwd()
        fname = ""
        if model == self.learned_patterns:
            fname = "learned_patterns"
        elif model == self.all_patterns:
            fname == "all_patterns"
        elif model == self.curr_patterns:
            fname = "current_patterns"
        fname = fname + ".pkl"
        fpath = os.path.join(base_path, fname)
        with open(fpath, "wb") as f:
            dill.dump(model, f)


    def find_exact_patterns(self, base_pattern):
        '''
        Checks if there are patterns in the current document that match base_pattern, a previously learned pattern.
        look by base patterns, as they don't have the variable/value
        :param base_pattern: a pattern object
        :return:
        '''
        if base_pattern in self.curr_patterns.keys():
            return self.curr_patterns[base_pattern]
        else: return None

    def find_close_patterns(self, base_pattern):
        '''
        TODO needs to check aliases
        Finds the hpattern in current_patterns that is closest to the hpattern that corresponds to the base_pattern.
        This is done by calculating the fuzzy distance between the entity name of the previously learned pattern
        and the entity name of all the patterns in current patterns. We also look for similar units. An overall
        confidence score of "closeness" is then calculated as the sum of close entity name parts and similar unit. So,
        a pattern in curr_patterns will have a higher confidence score if it is close to more than one entity previously
        learned than only one. The score is boosted by matching units, and the score is boosted if a multi-part entity
        name matches more than one part of a previously learned multi-part entity.
        :param pattern: pattern object
        :return close_patterns: dict, {pattern object: confidence score}.
        '''
        print("Trying to find a pattern close to ")
        print(base_pattern)
        #get the pattern_object corresponding to the base pattern we want to match
        pattern = self.all_patterns[base_pattern]
        hpattern = ast.literal_eval(str(pattern.hpattern))
        entities = pattern.find_entities()
        units = pattern.find_units()

        close_patterns = {} #pattern_object: confidence score

        #check if any of the patterns in the current document are close to hpattern
        for k, p in self.curr_patterns.items():
            #set defaults
            confidence_flag_entity = 0 #is the entity close?
            confidence_flag_unit = 0    #is the unit close?
            confidence = 0

            #get p's attributes
            p_hpattern = ast.literal_eval(str(p.hpattern))
            p_entities = p.find_entities()
            p_units = p.find_units()

            #score entity confidence by how many of the entity groups are close
            for p_ent in p_entities:
                for ent in entities:
                    #print("Checking the entity " + p_ent + " against the learned entity " + ent)
                    #print(fuzz.ratio(p_ent, ent))
                    if fuzz.ratio(p_ent, ent) > 70:
                        confidence_flag_entity += 1  # !Note: this is changed from the original code from a binary to a count. TODO: discuss with Asitang

            #score unit confidence by exact match
            for p_unit in p_units:
                for u in units:
                    #print("Checking the unit " + p_unit + " against the learned unit " + u)
                    if p_unit.lower() == u.lower(): #TODO make this more robust by using the GROBID json. Multiple representations can mean the same thing, like m and meter
                        confidence_flag_unit += .5 #weight a unit match less heavily than an entity match. #TODO evaluate different weights

                # ?? TODO Should we compare values too?
            confidence = confidence_flag_unit + confidence_flag_entity
            if confidence >= 1:
                if p not in close_patterns:
                    close_patterns[p] = 0
                close_patterns[p] += confidence

        return close_patterns

    def find_far_patterns(self, entity_name, aliases):
        """

        :param entity_name: string. The entity we want to find in the current document
        :param aliases: list of strings, Contains alternate representations of the desired entity
        :return:
        """
        aliases.append(entity_name)
        #print("aliases inside far patterns ", aliases)
        far_patterns = {}
        for alias in aliases:
           # print("finding patterns close to alias ", alias)
            for k, p in self.curr_patterns.items():
                entities = p.find_entities()
                for entity in entities:
                    score = fuzz.ratio(alias.lower(), entity.lower())
                    #print("alias and entity %s %s %d"  % (alias.lower(), entity.lower(), score))
                    if score > 50:
                        far_patterns[p] = score
        return far_patterns


    def matcher_bo_entity(self, entity_name, seed_aliases):
        '''
        When a new entity is processed, this function checks whether the entity has already
        been learned.
        If the entity name is already present in the learned_patterns, the base_patterns that have been learned
        for the entity are fetched. For instance, suppose we are processing the entity "Max Pressure" on a new
        document, and we have learned the pattern "max press @ num" from previous docs. We then check if this
        pattern exists in the current doc. If it does, retrieve the pattern. If the exact pattern does not exist
        in the current document, the function will find the closest pattern in the current document. For instance,
        perhaps instead of "max press @ num" the new document has "max pressure: num psi".


        Currently seed_aliases is add manually by the developer and is a comma separated string
        :param entity_name: string. The entity name to check if it has already been learned
        :return:
        '''
        exact_patterns = []
        close_patterns = []
        far_patterns = []
        #print(seed_aliases)
        # check if the any patterns for the entity have already been learned
        # TODO should we check in the seed aliases too?
        if entity_name in self.learned_patterns.keys():
            #get all the base_patterns we have already learned for the entity
            pre_learned_bpatterns = self.learned_patterns[entity_name]["base_patterns"]
            if pre_learned_bpatterns:
                # TODO set masks in the pattern object
                print('We have seen this entity before! Let us see if we can find an exact match in this document...')
                #check if we can find an exact match in the current document
                exact_patterns = list(map(self.find_exact_patterns, pre_learned_bpatterns))
                #filter any that returned None
                exact_patterns = [p for p in exact_patterns if p]
                if exact_patterns:
                    print('looks like the entity is present in the same form! Great!')
                else:
                    #look for patterns in the current document that are similar to the previously learned patterns
                    #TODO: what if there is an exact match and a close match? Could there be more than one representation?
                    print('No exact match found. Finding patterns in this document close to the learned patterns ...')
                    #print(pre_learned_bpatterns)
                    close_pats = list(map(self.find_close_patterns, pre_learned_bpatterns))
                    #close_patterns is a list of dictionaries. Key pattern, value confidence.
                    #  First aggregate keys into one big dictionary
                    close_p = {}
                    for d in close_pats:
                        for p, c in d.items():
                            if p not in close_p:
                                close_p[p] = 0
                            close_p[p] += c
                    # sort
                    close_p = sorted(close_p.items(), key=lambda x: x[1], reverse=True)
                    close_patterns = [tup[0] for tup in close_p]


        #if we haven't learned anything about this entity, check for patterns in the current document that are similar
        #to the entity name and seed_aliases
        #else:
        print("This entity hasn't been learned yet. Looking for patterns that are similar to the entity name.")
        print("Aliases before passing to far patterns ", seed_aliases)
        far_p = self.find_far_patterns(entity_name, seed_aliases)
        far_p = sorted(far_p.items(), key=lambda x: x[1], reverse = True)
        far_patterns = [tup[0] for tup in far_p]
        if len(far_patterns) > 30:
            far_patterns = far_patterns[0:30]

        return exact_patterns, close_patterns, far_patterns

    def matcher_bo_num(self, entity_value):
        """
        Searches for all the patterns in current_pattern that have the particular value associated with them
        :param entity_value: float
        :return: 
        """
        found_patterns = []
        for base_pattern, p in self.curr_patterns.items():
            if entity_value in p.all_nums:
                found_patterns.append(p)
        return found_patterns


    def update_learned_patterns(self, entity_name, patterns, aliases):
        """
        Adds the base_pattern of pattern to the learned patterns for the entity, entity_name
        :param entity_name: string
        :param pattern: pattern object or list of pattern objects
        :param aliases: dict. Each key is an entity name, values are lists of aliases for the entity.
        :return:
        """
        if isinstance(patterns, Pattern):
            patterns = [patterns]

        for pattern in patterns:
            #add the pattern to the entity if its not already there
            if entity_name not in self.learned_patterns:
                self.learned_patterns[entity_name] = {}
                self.learned_patterns["seed_aliases"] = []

            if pattern.base_pattern not in self.learned_patterns[entity_name]["base_patterns"]:
                self.learned_patterns[entity_name]["base_patterns"].append(pattern.base_pattern)

        #TODO decide how and if entity aliases should be saved.
        for ent, als in aliases.items():
            for alias in als:
                if als not in self.learned_patterns[entity_name]["seed_aliases"]:
                    self.learned_patterns[entity_name]["seed_aliases"].append(als)
        #save the model, in case of a crash or something
        self.save(self.learned_patterns)


    def make_report_by_page(self):
        """
        This is a mess and hopefully a soon to be deprecated function. Don't look at me with those judgemental eyes.
        :return:
        """
        report = {}
        docs_and_pages = {}
        for entity_name, values in self.learned_patterns.items():
            base_patterns = self.learned_patterns[entity_name]["base_patterns"]
            entity_report = {}
            for bp in base_patterns:
                #get the pattern object
                p = self.all_patterns[bp]
                pat_report = p.report()
                entity_report.update(pat_report)
            for doc, values in entity_report.items():
                if doc not in docs_and_pages:
                    docs_and_pages[doc] = []
                #the keys are all the page numbers
                docs_and_pages[doc].extend(list(entity_report[doc].keys()))
            #remove duplicates of pages
            for k,v in docs_and_pages.items():
                docs_and_pages[k] = list(set(docs_and_pages[k])) #TODO this is bad programming practice change this later
            report[entity_name] = entity_report


        #add all the entities to a row of the dataframe page by page, document by document
        all_rows = []
        for doc, pages in docs_and_pages.items():
            for page in pages:
                row = {"Document": doc}
                for entity_name, d in report.items():
                    #check if the document exists for this entity
                    if not doc in d.keys():
                        d[doc] = {}
                    #check if anything was found on the page for this entity
                    if page in d[doc].keys():
                        value_list = d[doc][page]
                        #convert the list of values to a string of line separated values
                        value_str = "\n".join(value_list)
                    else:
                        value_str = ""
                    row[entity_name] = value_str
                all_rows.append(row)
        df_report = pd.DataFrame(all_rows)
        #write to csv
        df_report.to_csv("report.csv", index=False)