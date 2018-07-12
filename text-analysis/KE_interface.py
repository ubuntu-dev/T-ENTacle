"""
TODO: write close function, put load and save in utils

Once a pattern has been learned, should it be removed from current patterns? For example,
if a user selects a pattern for "Average Pressure", and that pattern is left in current patterns,
it could be suggested again for something like "Max Pressure". This could be good in the case
of multiple possible matches, or bad if we desire uniqueness. There could be a flag to specify...
"""
import utils
import os
import pandas as pd
import numpy as np
from KE_class import KnowledgeExtractor
from pattern import Pattern
import locale
import ast

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
        print("trying to set entity seeds")
        if seed_path:
            with open(seed_path, "r") as f:
                seeds = f.readlines()
                entity_seeds = [name.strip() for name in seeds if name]
        else:
            print("no entity seeds were passed")
            entity_seeds = []
        return entity_seeds


    def get_user_selection(self, entity_name, patterns, select_mask=True):
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
            menu += str(count) + ". " + " ".join(p.instances[0]) + "\n"
            count += 1

        print("I found the following patterns that may represent " + entity_name + ".\n")
        print(menu)
        output = "Please enter the indices, separated by spaces, of all of the patterns you wish to select.\n" + \
                 "Or enter s to skip this entity or v to search by value."
        pattern_selection = input(output)
        if pattern_selection in ['v', 'V', 's', 'S']:
            return pattern_selection

        #TODO more user validation here. For now assume user behaves as expected
        pattern_selection = pattern_selection.split(" ")
        patterns_to_keep = [patterns[int(i)] for i in pattern_selection if i]

        for p in patterns_to_keep:
            print("You chose "+ " ".join(p.instances[0]))
            print("The value of select mask is ", select_mask)
            if select_mask:
            # for each pattern that the user selected, allow them to select the appropriate tokens (the mask).
                valid_input = False
                while not valid_input:
                    print(p.instances[0])
                    mask_message = "Please enter, separated by spaces, the indices of the pattern \"" + " ".join(p.instances[0]) + \
                                    "\" you wish to keep. Or enter a to keep all."
                    print(mask_message)
                    output = [str(i) + " " + p.instances[0][i] for i in range(len(p.instances[0]))]
                    output = "\t".join(output)
                    mask_selection = input(output)
                    if mask_selection == "a" or mask_selection == "A":
                        new_mask = list(range(len(p.instances[0])))
                        p.set_mask(new_mask)
                    else:
                        try:
                            mask_selection = mask_selection.split(" ")
                            mask_selection = [s for s in mask_selection if s]
                            new_mask = [int(i) for i in mask_selection]
                            if len(new_mask) <= len(p.instances[0]):
                                p.set_mask(new_mask)
                                valid_input = True
                            else:
                                print("You entered too many indices. Please try again.")
                        except ValueError:
                            print("I didn't understand your input. Please try again.")

            self.knowledge_extractor.update_learned_patterns(entity_name, p)
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
        print("far patterns")
        for p in far_patterns:
            print(p.base_pattern)

        #save the results if we have any, and poke the user if needed.
        print("strict ", strict)
        if strict == True:
            #do not interact with the user at all
            if len(exact_patterns) == 0:
                print('Since no exact pattern found, and interactive is False. Skipping this entity.')
                return
            else:
                print("Adding exact matches")
                #only add exact matches
                self.knowledge_extractor.update_learned_patterns(entity_name, exact_patterns)

        else:
            print("interactive mode")
            selection = None
            if auto_skip == True and len(exact_patterns) != 0:
                # don't bother the user for exact matches, just save
                print("auto skip is on")
                print("Found exact pattern")
                self.knowledge_extractor.update_learned_patterns(entity_name, exact_patterns)
            #get user input if we found close or far matches
            elif len(close_patterns) != 0:
                print("Found close patterns")
                #TODO should this be true?
                selection = self.get_user_selection(entity_name, close_patterns, False)
            elif len(far_patterns) != 0:
                print("Found far patterns")
                selection = self.get_user_selection(entity_name, far_patterns, True)

            else:
                print("I couldn't find anything that matched the entity. ")
                return

            #check if the user wants to skip this entity or search by value instead
            if selection == "s" or selection == "S":
                return
            if selection == "v" or selection == "V":
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

        :return:
        '''

        #maybe os dependent TODO change to some other way of getting doc name from file path
        doc_name = file_path.split("/")[-1]
        #parse the document for text with tika
        parsed_text = utils.parse_document(file_path)

        #this line for testing on a txt without tika
        # with open(file_path, "r") as f:
        #     parsed_text = f.readlines()
        
        #create knowledge extractor object and find all patterns
        self.knowledge_extractor.create_patterns_per_doc(parsed_text, doc_name)
        #TODO ? set all_patterns?

        if len(self.entity_seeds) > 0:
            for entity_name in self.entity_seeds:
                print('Processing the entity: '+entity_name)
                self.interact_for_single_entity(entity_name, strict=strict, auto_skip=auto_skip)

        #once finished generate a report of what has been learned.
        self.knowledge_extractor.make_report_by_page()

