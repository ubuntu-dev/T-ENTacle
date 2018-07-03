"""
TODO: write close function, put load and save in utils
"""
import utils
import os
import pandas as pd
import numpy as np
from KE_class import KnowledgeExtractor
from pattern import Pattern
import locale

class Interface:
    def __init__(self, seed_path):
        #set object vars
        self.entity_seeds = self.set_entity_seeds(seed_path)
        self.knowledge_extractor = KnowledgeExtractor()

    def set_entity_seeds(self, seed_path):
        """
        TODO add file validation with try except block
        Sets the entity seeds in the file seed_path.
        :param seed_path: string.
        :return:
        """
        if seed_path:
            with open(seed_path, "r") as f:
                seeds = f.readlines()
                self.entity_seeds = [name.strip() for name in seeds if name]
        else:
            self.entity_seeds = []


    def get_user_selection(self, entity_name, patterns, knowledge_extractor, select_mask=True):
        """
        Allows the user to select the best matching patterns for the entity, entity_name. After selecting the patterns,
        the user can also select to only keep part of the pattern.
        :param entity_name: string, the entity_name to find patterns for
        :param patterns: pattern object or list of pattern objects.
        :param knowledge_extractor: KnowledgeExtractor object.
        :param select_mask: bool. Allows the user to select only parts of the pattern (or not).
        :return: None or string. A string is returned if the user elects to break out of this process.
        """
        #check for correct data type
        if isinstance(patterns, Pattern):
            patterns = [patterns]

        #create menu and print prompt for user
        count = 0
        menu = ""
        for p in patterns:
            menu = str(count) + ". " + p.instances[0] + "\n"
            count += 1

        print("I found the following patterns that may represent " + entity_name + ".\n")
        print(menu)
        output = \
        """Please enter the indices, separated by spaces, of all of the patterns you wish to select.
        Or enter s to skip this entity or v to search by value."""
        pattern_selection = input(output)
        if pattern_selection in ['v', 'V', 's', 'S']:
            return pattern_selection

        patterns_to_keep = [patterns[i] for i in pattern_selection]

        for p in patterns_to_keep:
            if select_mask:
            # for each pattern that the user selected, allow them to select the appropriate tokens (the mask).
                valid_input = False
                while not valid_input:
                    mask_message = "Please enter, separated by spaces, the indices of the pattern " + p.instances[0] + \
                                    " you wish to keep. Or enter a to keep all."
                    print(mask_message)
                    output = [str(i) + " " + p.instances[0][i] for i in range(len(p.instances[0]))]
                    output = "\t".join(output)
                    mask_selection = input(output)
                    if mask_selection == "a" or mask_selection == "A":
                        new_mask = list(range(len(p.instances[0])))
                        p.set_mask(new_mask)
                    else:
                        try:
                            new_mask = [int(i) for i in mask_selection]
                            if len(new_mask) <= len(p.instances[0]):
                                p.set_mask(new_mask)
                                valid_input = True
                            else:
                                print("You entered too many indices. Please try again.")
                        except ValueError:
                            print("I didn't understand your input. Please try again.")

            knowledge_extractor.update_learned_patterns(p)
        return None

    def get_value_from_user(self):
        value_prompt = "Please enter the value you would like to search for. I will find a pattern that contains the value."
        value = input(value_prompt)
        return value

    def interact_for_single_entity(self, entity_name, strict=False, auto_skip=False):
        '''
        TODO rewrite this entire function
        1. look for exact/similar entities
        2. if strict is True and exact found, then update the learned
        3. Else, don't bother the user
        4. just skip that entity, put it in the report
        :param knowledge_extractor: KnowledgeExtractor object.
        :param entity_name: string.
        :param strict: bool. If True specifies that the user wants no interaction. Will only save exact pattern matches.
        :param auto_skip: bool. Specifies partial supervision. If true, exact patterns will be saved without user feedback,
                                but the program will prompt the user for feedback on similar pattern matches.
        :return:
        '''
        #first check if we have already learned the entity or something similar to it
        exact_patterns, close_patterns, far_patterns = self.knowledge_extractor.matcher_bo_entity(entity_name)

        #save the results if we have any, and poke the user if needed.
        if strict == True:
            #do not interact with the user at all
            if len(exact_patterns) == 0:
                print('Since no exact pattern found, and interactive is False. Skipping this entity.')
                return
            else:
                #only add exact matches
                self.knowledge_extractor.update_learned_patterns(entity_name, exact_patterns)

        else:
            if auto_skip == True:
                change_search = None
                # don't bother the user for exact matches, just save
                if len(exact_patterns) != 0:
                    self.knowledge_extractor.update_learned_patterns(entity_name, exact_patterns)
                #get user input if we found close or far matches
                elif len(close_patterns) != 0:
                    change_search = self.get_user_selection(entity_name, close_patterns, False)
                elif len(far_patterns) != 0:
                    change_search = self.get_user_selection(entity_name, far_patterns, False)

                #check if the user wants to skip this entity or search by value instead
                if change_search == "s" or change_search == "S":
                    return
                if change_search == "v" or change_search == "V":
                    #TODO right now, only searches for nums, maybe we want to allow to search by date etc
                    value = self.get_value_from_user()
                    try:
                        value_as_float = locale.atof(value)
                        value_patterns = self.knowledge_extractor.matcher_bo_num(value_as_float)
                        if value_patterns:
                            self.get_user_selection(entity_name, value_patterns)

                    except ValueError:
                        #TODO better error handling
                        print("Sorry, I didn't understand that input. ")


    def single_doc_cli(self, file_path, strict=False, auto_skip=False):
        '''
        controls the flow of the CLI
        :param file_path: string. The file path of a document to process
        :param knowledge_extractor: KnowledgeExtractor object. This object will process all the
                                    documents.
        :return:
        '''

        #init(file_path)
        #parse the document for text with tika
        parsed_text = utils.parse_document(file_path)
        #create knowledge extractor object and find all patterns
        ke = KnowledgeExtractor()
        ke.create_patterns_per_doc(parsed_text)
        #TODO ? set all_patterns?

        if len(self.entity_seeds) > 0:
            for entity_name in self.entity_seeds:
                print('Processing the entity: '+entity_name)
                self.interact_for_single_entity(entity_name, strict=strict, auto_skip=auto_skip)
        #      close()
        # else:
        #     continue_cli = True
        #     while continue_cli:
        #         entity_name = input('Enter the entity you\'d like to search:')
        #         self.interact_for_single_entity(entity_name, ke)
        #         decision=input('Do you want to continue searching: y or n')
        #         if decision=='n':
        #             continue_cli=False
        #             close()
        #

