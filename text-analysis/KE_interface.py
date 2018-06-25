"""
TODO: write close function, put load and save in utils
"""
import utils
import os
import pandas as pd
import numpy as np
import click
from KE_class import KnowledgeExtractor

class KE_Interface:
    def __init__(self):
        #set object vars
        self.entity_seeds = [] #list of entities to look for
        #set up a counter to save state across documents
        if os.path.exists('counter'):
            self.counter = self.loadmodel('counter')
        else:
            self.counter = 0

        # entity matchings for all the documents processed so far
        if os.path.exists("learned_patterns.csv"):
            self.learned_patterns = pd.read_csv('learned_patterns.csv', index_col=0) #I dropped the index on current patterns. Double check if I needed it for this to work.
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

    def interact_for_single_entity(entity_name, knowledge_extractor, strict=False, auto_skip=False):
        '''
        TODO rewrite this entire function
        1. look for exact/similar entities
        2. if strict is True and exact found, then update the learned
        3. Else, don't bother the user
        4. just skip that entity, put it in the report
        :param knowledge_extractor: KnowledgeExtractor object.
        :param entity_name: string.
        :param strict: bool.
        :return:
        '''

        result_rows = pd.DataFrame(columns=['instances', 'pattern_ids', 'hpattern'])
        exact_pattern_ids, close_pattern_ids, far_pattern_ids, exact_masks = knowledge_extractor.matcher_bo_entity(entity_name)

        if len(exact_pattern_ids) != 0:
            result_rows = get_rows_from_ids(exact_pattern_ids)
        elif len(close_pattern_ids) != 0:
            result_rows = get_rows_from_ids(close_pattern_ids)
        elif len(far_pattern_ids) != 0:
            result_rows = get_rows_from_ids(far_pattern_ids)
        else:
            print('Looks like there is nothing to be found here!')

        if strict == True:
            if len(exact_pattern_ids) == 0:
                print('since no exact pattern found, and interactive is False. Skipping this entity.')
                return
            else:
                for _, row in result_rows.iterrows():
                    pattern_id = row['pattern_id']
                    mask = exact_masks[pattern_id]
                    update_learn_pattern(entity_name, pattern_id, mask)

        else:
            if auto_skip == True and len(exact_pattern_ids) != 0:
                for _, row in result_rows.iterrows():
                    pattern_id = row['pattern_id']
                    mask = exact_masks[pattern_id]
                    update_learn_pattern(entity_name, pattern_id, mask)
                return

            output = ''
            count = 0
            id_map = ['dummy']
            for _, row in result_rows.iterrows():
                count += 1
                output += str(count) + '. ' + ' '.join(ast.literal_eval(str(row['instances']))[0]) + '\n'
                id_map.append(row['pattern_id'])

            output += 'ENTER s TO SKIP THIS ENTITY\n'
            output += 'ENTER v search the patterns by Value\n'

            print(
                'Please, enter the index corresponding to the instance that best represents what you are looking for:')
            selected_pattern_id = input(output)

            if selected_pattern_id == 's':
                return

            if selected_pattern_id == 'v':
                value = input('give the value for the entity: ' + entity_name)
                exact_pattern_ids, instance_samples = matcher_bo_value(value)
                output = ''
                count = 0
                id_map = ['dummy']
                for pattern_id, instance in zip(exact_pattern_ids, instance_samples):
                    count += 1
                    output += str(count) + '. ' + ' '.join(instance) + '\n'
                    id_map.append(pattern_id)

                output += 'ENTER s TO SKIP THIS ENTITY\n'
                if selected_pattern_id == 's':
                    return

                print(
                    'Please, enter the index corresponding to the instance that best represents what you are looking for:')
                selected_pattern_id = input(output)

                selected_instance = instance_samples[int(selected_pattern_id) - 1]

            selected_instance = ast.literal_eval(
                str(list(result_rows[result_rows['pattern_id'] == id_map[int(selected_pattern_id)]]['instances'])[0]))[
                0]

            output = ''
            count = 0
            for token in selected_instance:
                count += 1
                output += str(count) + '. ' + token + '    '
            print(
                'Please, enter (seperated by spaces) all the indexes of the tokens that are relevant to the entity that you are looking for:')
            selected_tokens = input(output)
            selected_tokens = selected_tokens.strip().split(' ')
            selected_tokens = list(map(int, selected_tokens))
            mask = []
            for token_id in range(1, len(selected_instance) + 1):
                if token_id in selected_tokens:
                    mask.append(True)
                else:
                    mask.append(False)

            update_learn_pattern(entity_name, id_map[int(selected_pattern_id)], mask)


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

        if len(self.entity_seeds)>0:
             for entity_name in self.entity_seeds:
                print('Processing the entity: '+entity_name)
                self.interact_for_single_entity(entity_name, ke, strict=strict, auto_skip=auto_skip)
             close()
        else:
            continue_cli = True
            while continue_cli:
                entity_name = input('Enter the entity you\'d like to search:')
                self.interact_for_single_entity(entity_name, ke)
                decision=input('Do you want to continue searching: y or n')
                if decision=='n':
                    continue_cli=False
                    close()

    @click.command()
    @click.argument('path')
    @click.option('--i/--no-i', default=True, help='choose --i for interactive')
    @click.option('--a/--no-a', default=False,
                  help='choose --a to auto skip exact patterns when in interactive mode. False by default.')
    @click.option('--f/--no-f', default=False,
                  help='choose --f to start fresh (all previous learning will be removed)')
    @click.option('--e', help='give the path to the list of entities. One entity per line.')
    def cli(self, path, i, a, f, e=''):
        '''
            The main function that drives the command line interface
            detect if dir or file and accordingly iterates or not over multiple docs
            :param file_path:
            :return:
            '''

        #if a fresh start is indicated, then remove all previous learning
        if f:
            utils.remove_files(['learned_patterns.csv','all_patterns.csv','current_patterns.csv','counter','report.csv'])

        strict=not i

        #load the seed entities
        if e!='':
            with open(e, "r") as f:
                self.entity_seeds = f.readlines()
            self.entity_seeds = [name.strip() for name in self.entity_seeds]
        else:
            self.entity_seeds=[]

        #TODO decide if each document should have its own KE object, or if it should share. prob its own??

        #read the files
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    #TODO: make checking for correct file types more robust
                    if file == '.DS_Store':
                        continue
                    # initialize a knowledge extractor
                    self.single_doc_cli(os.path.join(root, file), self.entity_seeds, strict, a)
        elif os.path.isfile(path):

            self.single_doc_cli(path, self.entity_seeds, strict, a)

