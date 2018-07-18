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
    TIME = 'Time~'
    ALPHANUM = 'AlphaNumeric'
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

    def __init__(self, tokens, page_num, doc_name, position):
        self.instance = tokens #list of strings
        self.page_num = page_num
        self.hpattern = self._create_hpattern(self.instance)
        self.base_pattern = self.get_base_pattern(self.hpattern)
        self.instances = [tokens]
        self.page_nums = [page_num]
        self.doc_name = doc_name
        #TODO add order, just the relative order that each instance has appeared. So instances and order will be the same length
        #in KE class, aggregate locations by document
        self.location = {doc_name: {"page_num": [page_num], "instances": [tokens], "order": [position]}} #this could eventually be some sort of character position for grouping
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
        """
        TODO make date recognition more robust
        :param token:
        :return:
        """
        if re.search(r"\d{1,2}\/\d{1,2}\/\d{2,4}", token):
            return True
        else:
            return False

    def is_time(self, token):
        """
        TODO unique to locale, make more robust
        assumes token already has_numeric
        :param token:
        :return:
        """
        if re.search(r":", token):
            return True
        else:
            return False

    def has_numeric(self, token):
        if re.search(r"\d", token):
            return True
        else:
            return False

    def has_alpha(self, token):
        if re.search(r"[a-zA-Z]", token):
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
                signature.append((token, Pattern.DATE, Pattern.DATE))
            elif self.has_numeric(token):
                if self.has_alpha(token):
                    signature.append((token, Pattern.ALPHANUM, Pattern.ALPHANUM))
                elif self.is_time(token):
                    signature.append((token, Pattern.TIME, Pattern.TIME))
                else:
                    signature.append((token, Pattern.DIGI, Pattern.DIGI))
            # elif self.is_punc(token):
            #     signature.append((token, token, Pattern.PUNC))
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
                    #TODO ask Asitang what the goal was here. This seems to break things for me
                    #signature.append(tuple(token))
                    #rewrite as
                    signature.append((token, token, token))

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
           # self.all_nums.sort()
        elif isinstance(new_instance, list):
            self.instances.extend(new_instance)
            self.all_nums.extend(self.convert_to_num(new_instance))
            #self.all_nums.sort()
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
                self.location[key]["order"].extend(location_dict[key]["order"])
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

    def find_values(self, instances = None):
        """
        Return the numerical part(s) of the pattern
        :return: values, list of strings representing numbers
        """
        if not instances:
            instances = self.instances
        indices = []
        for i in range(len(self.base_pattern)):
            if i in self.mask and self.base_pattern[i] == self.DIGI:
                indices.append(i)
        values = [" ".join([inst[i] for i in indices]) for inst in instances]
        return values

    def report(self):
        """
        aggregate all the instances by document and then by page
        report is in form {docname: {page1 : "val1\nval2\nval3"}, {page2 : "val1\nval2\nval3"}}
        :return:
        """
        report = {}

        for doc, dct in self.location.items():
            report[doc] = {}
            values = self.find_values(dct["instances"])
            for tup in zip(dct["page_num"], values):
                if tup[0] not in report[doc]:
                    report[doc][tup[0]] = []
                report[doc][tup[0]].append(tup[1])
        return report

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
                   'base_pattern': self.base_pattern, 'document_name': self.doc_name, 'num_instances': len(self.instances)}
        return as_dict

    def _convert_to_num_helper(self, instance, indices, as_num):
        """
        TODO this doesn't really work because some numbers are not really values like 25.789.67
        :param instance:
        :param indices:
        :param as_num:
        :return:
        """
        for i in indices:
            number = instance[i]
            try:
                # check if the number is a fraction:
                if "/" in number:
                    nums = number.split('/')
                    if len(nums) > 2:  # then it's not a fraction!
                        continue
                    else:
                        number = int(nums[0]) / int(nums[1])
                        as_num.append(number)
                else:
                    as_num.append(locale.atof(number))
            except ValueError:
                pass
                # print(ValueError)
                # print(self.hpattern)
                # print("index " + str(i))
                # print(instance)
                # print(number)
        return as_num

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
            if isinstance(instances, list):
                for instance in instances:
                    as_num = self._convert_to_num_helper(instance, indices, as_num)
            elif isinstance(instances, tuple):
                as_num = self._convert_to_num_helper(instances, indices, as_num)
            else:
                raise ValueError
                print("Expected type list or tuple but got type " + type(instances))

        as_num.sort()
        return as_num


