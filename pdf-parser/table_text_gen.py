#!/usr/bin/env python
import os
import glob
import sys
from random import shuffle


def get_random_words():
	"""
	Gets random words acquired at http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain,
	shuffles them and returns them as a list
    :return list_of_random_words list of shuffled words
	"""
	filepath = './words.txt'
	list_of_random_words = []  
	with open(filepath) as fp:
		list_of_random_words = fp.readlines()
	list_of_random_words = [w.strip() for w in list_of_random_words] 
	shuffle(list_of_random_words)
	return list_of_random_words


def read_words(words_file):
	"""
	Reads a text file and puts its words into a list
	:param words_file text file to be read
    :return list of words read from a file
	"""
    return [word for line in open(words_file, 'r') for word in line.split()]


def get_tech_words():
	"""
	Gets words from all the parsed PDFs files and returns them all in one list
    :return list_of_tech_words list of words extracted from the technical PDFs
	"""
	path = './XML_Stage2/*'   
	files = glob.glob(path)
	list_of_tech_words = []
	for name in files:
		words = read_words(name)
		for w in words:
			list_of_tech_words.append(w)
	return list_of_tech_words


def prepare_all_words():
	"""
	Gathers the two lists of words - random and tech - and shuffles them, followed by
	a creation of one final file of shuffled words, ready to populate tables
	"""
	list_of_random_words = get_random_words()
	list_of_tech_words = get_tech_words()
	combined_list = list_of_random_words + list_of_tech_words
	shuffle(combined_list)
	with open('text_for_tables.txt', 'w') as output:
		for w in combined_list:
			output.write(w + ' ')

prepare_all_words()
