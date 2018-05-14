import re
import spacy
from fuzzywuzzy import fuzz
import itertools
from text_container import TextContainer
import pandas as pd
from utils import Incrementer
from KVpairs import KVpairs
class EntityLabeler(object):

    def __init__(self, raw_text, unit_entities):
        """
        :param raw_text: string, raw text
        :param annotations: json, in grobid-quantities format.
        :param entity_units: a dictionary of unit: [entity_names_having_that_unit]
        """
        self.raw_text = raw_text
        self.unit_entities = self._set_unit_entities(unit_entities)
        self.spacy_parser = spacy.load('en')
        self.text_lines = self.raw_text.splitlines()

    def _set_unit_entities(self, unit_ent):
        for k,v in unit_ent.items():
            for word in v:
                multiword = word.split()
                if len(multiword) > 1:
                    acronym = [w[0] for w in multiword]
                    acronym = "".join(acronym)
                    unit_ent[k].append(acronym)
        return unit_ent

    def _split_text(self):
        text_lines = self.raw_text.splitlines(True)
        return txt

    def find_nums(self, text):

        parsed = self.spacy_parser(text)
        nums = []
        for ent in parsed.ents:
            if ent.label_ == "QUANTITY":
                print(parsed)
                #get tokens to left
                left = text[0:ent.start_char-1]
                print(left)
                left_tokens = [token for token in self.spacy_parser(left)][-3:]
                right = text[ent.end_char+1:]
                right_tokens = [token for token in self.spacy_parser(right)][0:2]
                nums.append((left_tokens, ent.text, right_tokens))
        return nums

    def parse(self):
        for line in self.text_lines:
            nums = self.find_nums(line)
            #print(nums)