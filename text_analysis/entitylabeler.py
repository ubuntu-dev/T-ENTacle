##assume we have some text in a file or something

import re
import spacy
from fuzzywuzzy import fuzz
import itertools
from text_container import TextContainer
import pandas as pd
from utils import Incrementer
from KVpairs import KVpairs
from copy import deepcopy

class EntityLabeler(object):

    def __init__(self, raw_text, annotations, unit_entities):
        """
        :param raw_text: string, raw text
        :param annotations: json, in grobid-quantities format.
        :param entity_units: a dictionary of unit: [entity_names_having_that_unit]
        """
        self.raw_text = raw_text
        self.annotations = annotations
        self.unit_entities = self._set_unit_entities(unit_entities)
        self.spacy_parser = spacy.load('en')
        #self.annotations_alias = self._get_quant_list()
        self.text_lines = self._split_text()
        #self.report = pd.DataFrame(columns = self.unit_entities.values())
        self.records = {}
        self.new_annotations = []
        self.annot_incrementer = Incrementer(self.annotations)

    # def _split_text(self):
    #     text_lines = []
    #     for m in re.finditer("\n", self.raw_text):
    #         if m:
    #             print(m.start())
    #             print(m.end())
    #             print(self.raw_text[m.start(), m.end()])
    #             text_lines.append(TextContainer(self.raw_text[m.start():m.end()], m.start(), m.end()))
    #     return text_lines
    def _split_text(self):
        text_lines = self.raw_text.splitlines(True)
        start = 0
        end = 0
        txt = []
        for line in text_lines:
            end += len(line)
            txt.append(TextContainer(line, start, end))
            start = end
        return txt


    def _set_unit_entities(self, unit_ent):
        for k,v in unit_ent.items():
            for word in v:
                multiword = word.split()
                if len(multiword) > 1:
                    acronym = [w[0] for w in multiword]
                    acronym = "".join(acronym)
                    unit_ent[k].append(acronym)
        return unit_ent

    def _get_quant_list(self):
        quants = []
        for item in self.annotations:
            for key, val in item.items():
                if key.startswith("quantity"):
                    quants.append(item[key])
        return quants

    def _get_candidate_entities(self, unit):
        if unit in self.unit_entities:
            return self.unit_entities[unit]
        else: return None


    def get_text_left(self, offset, n):
        #tokenize the text to the right and grab n words. n tbd
        right_text = self.raw_text[0:offset[0]+1]
        parsed = self.spacy_parser(right_text)
        n_rights = [token.text.lower() for token in parsed if token.is_alpha][-n:]
        return n_rights


    def get_text_right(self):
        """
        something to think about. Maybe the quantified thingy is to the right!
        :return:
        """

    def powerset(self, ls):
        pset_tuples = itertools.chain.from_iterable(itertools.combinations(ls,r) for r in range(1,len(ls)+1))
        pset_strings = [" ".join(tup) for tup in pset_tuples]
        return pset_strings

    def _get_most_similar_ent(self, candidate_ents, words):
        """

        :param candidate_ents: list, a list of entities having the same unit as the value
        :param words: list, the word tokens preceding the value
        :return: tuple of strings, (candidate entity having the highest Levenshtein value with words, the token(s) preceding
                                    the number, i.e. the thing being quantified)
                                    TODO needs to return offsets for labeling
        """
        #start with levenshtein and then (based on performance) move to something more sophisticated like a word2vec model
        all_word_combos = self.powerset(words)
        pair_tuples = list(itertools.product(candidate_ents, all_word_combos))
        pair_lists = list(zip(*pair_tuples))
        scores = list(map(fuzz.ratio, pair_lists[0], pair_lists[1]))
        pair_scores = zip(pair_tuples, scores)
        pair_scores = sorted(pair_scores, key = lambda x: x[1], reverse = True)
        best_score = pair_scores[0]
        #cutoff value
        if best_score[1] > 80:
            return best_score[0]
        else:
            return None
        #TODO determine a cutoff, return highest val or None

    def _entities_from_grobid(self, raw_text, annotation):
        if re.search('[a-zA-Z]', annotation["rawValue"]):
            #Grobid fucked up. return and find entities with regex
            return None
            #TODO add relabeling to retrain Grobid
        else:
            if "rawUnit" in annotation:
                #get candidate entities
                unit = annotation["rawUnit"]["name"].lower()
                candidates = self._get_candidate_entities(unit)
                #TODO this is a hack in case grobid got the wrong unit. Fix
                if not candidates:
                    candidates = list(itertools.chain.from_iterable([v for k, v in self.unit_entities.items()]))
            else:
                candidates = list(itertools.chain.from_iterable([v for k, v in self.unit_entities.items()]))

            if candidates is not None:
                candidates = [w.lower() for w in candidates]
                #get the preceding words
                words = self.get_text_left((annotation["offsetStart"], annotation["offsetEnd"]), 4)
                if words:
                    #calculate word similarity
                    entity = self._get_most_similar_ent(candidates, words)
                    return entity
                    #TODO probably need a way of getting its offsets for retraining GQ
            else: #no candidates in list, needs a new type added
                pass
                #TODO add user interaction loop to label new type
                #unit could be wrong, check against all entities


    def _entities_from_regex(self, raw_text):
        regex_finder = KVpairs(raw_text)

        return regex_finder.get_entities()

    def _flatten(self):
        if self.records:
            record = {key: ["\n".join(value)] for key, value in self.records.items()}
            return record
            #return record

    def _set_quant_key(self, annotation):
        quant_key = [k for k in annotation.keys() if k.startswith("quantity") or k == "quantities"][0]
        return quant_key

    def _set_regex_entities(self, regex_entities):
        if regex_entities:
            # loop through the list of singleton dictionaries and
            # add any found keys
            for key, val in regex_entities.items():
                if key not in self.records:
                    self.records[key] = []
                self.records[key].append(val)


    def _find_entities(self, line, quantity_annotation):
        """
        Checks for a Grobid Quant, and if none, checks for a regex quantity
        :param line:
        :param quantity_annotation:
        :return:
        """
        entity_tuple = self._entities_from_grobid(line.text, quantity_annotation)
        if entity_tuple:
            # relabel with quantified substance for retraining grobid
            new_annot = deepcopy(quantity_annotation)
            new_annot["quantified"] = {"rawName": entity_tuple[1]}  # TODO add offsets
            #TODO add pipeline for retraining grobid
            # add entity and value to records
            entity_name = entity_tuple[0]
            if entity_name not in self.records:
                self.records[entity_name] = []
            self.records[entity_name].append(quantity_annotation["rawValue"])
        # get next annot
        else:
            # try with regex...
            regex_entities = self._entities_from_regex(line.text)
            self._set_regex_entities(regex_entities)

    def traverse_text_and_annotations(self, line, quantity_annotation, current_annot):
        if line.end < quantity_annotation["offsetStart"]:
            # use regex to find some stuff
            regex_entities = self._entities_from_regex(line.text)
            self._set_regex_entities(regex_entities)

        else:
            while line.end >= quantity_annotation["offsetStart"]:
                self._find_entities(line, quantity_annotation)
                if self.annot_incrementer.has_next():
                    current_annot = self.annot_incrementer.increment()
                    quant_key = self._set_quant_key(current_annot)
                    if quant_key == "quantities":
                        for item in current_annot[quant_key]:
                            quantity_annotation = item
                            if line.end >= quantity_annotation["offsetStart"]:
                                self._find_entities(line, quantity_annotation)
                            else:
                                break
                    else:
                        quantity_annotation = current_annot[quant_key]
                else:
                    break

        return current_annot

    def parse(self):
        """
        loop through the lines and the annotations
        """
        annot_incrementer = Incrementer(self.annotations)
        current_annot= annot_incrementer.set()
        #check if Grobid-Q has labeled anything in this line
        for line in self.text_lines:
            if line.text.isspace():
                continue
            quant_key = self._set_quant_key(current_annot)
            if quant_key == "quantities":
                for item in current_annot[quant_key]:
                    current_annot = self.traverse_text_and_annotations(line, item, current_annot)

            else:
                current_annot = self.traverse_text_and_annotations(line, current_annot[quant_key], current_annot)
        return self._flatten()

    def parse_no_annot(self):
        for line in self.text_lines:
            regex_ents = self._entities_from_regex(line.text)
            self._set_regex_entities(regex_ents)
        return self._flatten()