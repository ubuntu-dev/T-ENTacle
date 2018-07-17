import click
import utils
import os
from KE_interface import Interface

@click.command()
@click.argument('path')
@click.option('--i/--no-i', default=False, help='choose --i for interactive')
@click.option('--a/--no-a', default=False,
              help='choose --a to auto skip exact patterns when in interactive mode. False by default.')
@click.option('--f/--no-f', default=False,
              help='choose --f to start fresh (all previous learning will be removed)')
@click.option('--e', help='give the path to the list of entities. One entity per line.')
#@click.option('--aliases', default=None, help='path to entity aliases. Comma separated per line')
def cli(path, i, a, f, e=''):
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
    #TODO decide if each document should have its own KE object, or if it should share. prob its own??

    #read the files
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                #TODO: make checking for correct file types more robust
                if file == '.DS_Store':
                    continue
                # initialize a knowledge extractor
                interface = Interface(e)
                interface.single_doc_cli(os.path.join(root, file), strict, a)
    elif os.path.isfile(path):
        interface = Interface(e)
        interface.single_doc_cli(path, strict, a)


if __name__ == "__main__":
    cli()
