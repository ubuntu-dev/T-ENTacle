import string
import ast
import re
from itertools import chain
import locale

class Pattern(object):
    PREP = "Prep~"
    PUNC = 'Punc~'
    WORD = 'Word~'
    DIGI = 'Digi~'
    UNIT = 'Unit~'
    DATE = 'Date~'
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
    # could be made more robust by taking the list of units from grobid. This is certainly not comprehensive
    units = ['ft', 'gal', 'ppa', 'psi', 'lbs', 'lb', 'bpm', 'bbls', 'bbl', '\'', "\"", "'", "°", "$", 'hrs']
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    def __init__(self, tokens, page_num, doc_name, id):
        print(tokens)
        self.instance = tokens
        self.page_num = page_num
        self.hpattern = self._create_hpattern(self.instance)
        self.base_pattern = self.get_base_pattern(self.hpattern)
        self.instances = [tokens]
        self.page_nums = [page_num]
        self.doc_name = doc_name
        self.id = id
        self.location = {doc_name: {"page_num": [page_num], "instances": [tokens]}} #this could eventually be some sort of character position for grouping
        self.mask = range(len(tokens))
        self.all_nums = self.convert_to_num(self.instances)

    def set_mask(self, mask):
        """
        The parts of the pattern the user chose
        :param mask: List of indices
        :return:
        """
        self.mask = mask

    def is_date(self, token):
        if re.search(r"\\\d{1,2}\\\d{1,2}\\\d{2,4}", token):
            return True
        else:
            return False

    def has_numeric(self, token):
        if re.search(r"\d", token):
            return True
        else:
            return False


    def is_punc(self, s):
        if re.match('[^a-zA-Z\d]', s):
            return True
        return False

    def _create_hpattern(self, instance):
        '''
        creates a heirarchy of 'denominations/classes' for each base pattern
        :param instance: list, tokenized string
        :return: base_pattern, h_pattern
        '''

        signature = []
        #print(instance)
        for token in instance:
            #print(token)
            if token in Pattern.prepos:
                signature.append((token, token, Pattern.PREP))
            elif self.is_date(token):
                signature.append(token, Pattern.DATE, Pattern.DATE)
            elif self.has_numeric(token):
                signature.append((token, Pattern.DIGI, Pattern.DIGI))
            elif token.isalpha():
                sign = [token, token, Pattern.WORD]
                if token.lower() in Pattern.units:
                    sign.append(Pattern.UNIT)
                signature.append(tuple(sign))

            #maybe use spacy or nltk instead of ispunc
            elif self.is_punc(token):
                signature.append((token, token, Pattern.PUNC))
            else:
                if token:
                    signature.append(tuple(token))

        return tuple(signature)

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

    def add_instance(self, new_instance):
        """
        Adds another instance of the pattern
        :param new_instance: tuple of strings or list of tuples of strings
        :return:
        """
        if isinstance(new_instance, tuple):
            self.instances.append(new_instance)
            self.all_nums.append(self.convert_to_num(new_instance))
            self.all_nums.sort()
        elif isinstance(new_instance, list):
            self.instances.extend(new_instance)
            self.all_nums.extend(self.convert_to_num(new_instance))
            self.all_nums.sort()
        else:
            error_message = 'Expected a string or list but got type ' + str(type(new_instance))
            raise TypeError(error_message)

    def add(self, location_dict):
        """
        Adds more findings of the pattern
        :param location_dict: nested dictionary of form {"doc_name" : {"page_num" :[], "instances": []}}
        :return:
        """
        for key in location_dict.keys():
            if key in self.location:
                self.location[key]["page_num"].extend(location_dict[key]["page_num"])
                self.location[key]["instances"].extend(location_dict[key]["instances"])
            else:
                self.location[key] = location_dict[key]
            self.page_nums.extend(location_dict[key]["instances"])
            self.instances.extend(location_dict[key]["instances"])

    def add_page_num(self, new_page_num):
        self.page_nums.append(new_page_num)

    def find_entities(self):
        """
        Aggregates the part of the hpattern that are words to find the entire entity
        :return:
        """
        entity = ""
        entities = []
        for token in [self.hpattern[i] for i in self.mask]:
            if token[2] == self.WORD and len(token) < 4: #don't include units in the entity name
                entity += token[0] + " "
            else:
                if entity != "":
                    entities.append(entity.strip())
                    entity = ""
        return entities

    def find_units(self):
        """
        Return the unit portion of the pattern
        :return: units, list of strings representing a unit
        """
        units = []
        for token in [self.hpattern[i] for i in self.mask]:
            if len(token) >= 4 and token[3] == self.UNIT:
                units.append(token[0])
        return units

    def find_values(self):
        """
        Return the numerical part(s) of the pattern
        :return: values, list of strings representing numbers
        """
        values = []
        for token in [self.hpattern[i] for i in self.mask]:
            if token[3] == self.DIGI:
                values.append(token[0])
        return values

    def get_string(self):
        """
        Represents the objects variables in a nice little sentence
        :return: as_string, string
        """
        ins = " ".join(self.instance)
        bp = " ".join(self.base_pattern)
        flat_h = []
        for p in self.hpattern: flat_h += p
        c = " ".join(flat_h)
        as_string = "instance: " + ins + "\n base_pattern: " + bp + "\n hpattern: " + c + "\n page num: " + str(self.page_num)
        return as_string

    def get_dict(self):
        as_dict = {'instances': self.instances, 'page_numbers': self.page_nums, 'hpattern': self.hpattern,
                   'base_pattern': self.base_pattern, 'doc_name': self.doc_name}
        return as_dict

    def convert_to_num(self, instances):
        """
        Stores the numeric part of an instance as an instance and keeps in sorted order
        TODO: use a better data structure for sorting, like a linked list or ordered dict or tree!
        :param instances:
        :return:
        """
        as_num = []
        if Pattern.DIGI in self.base_pattern:
            indices = [i for i in range(len(self.base_pattern)) if self.base_pattern[i] == Pattern.DIGI]
            for instance in instances:
                for i in indices:
                    as_num.append(locale.atof(instance[i]))
        as_num.sort()
        return as_num


