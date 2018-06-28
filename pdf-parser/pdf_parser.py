#!/usr/bin/env python
import tika
from tika import parser
import os
import xml.etree.ElementTree as ET
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.tag import StanfordNERTagger

def parse_pdf():
	"""
	Reads all PDF files and generates an XML files for each using Tika
	"""
	i = 0
	for x in os.listdir("./PDF/"):
		if './PDF/' + str(x) == '.DS_Store':
			print("SKIP")
			continue
		parsed = parser.from_file('./PDF/' + str(x), xmlContent=True)
		filename = x[:len(x)-5]
		i = i + 1
		with open('./XML/' + filename + '.xml', 'w') as output:
			print(filename)
			output.write(parsed["content"])

def remove_names(sentence):
	"""
	Removes NNPs (proper nouns) from a sentence
	:param sentence the sentence to be modified
    :return the de-identified/modified sentence
	"""
	new_sent = sentence
	for w in sentence.split():
		if '/' in w:
			replacement = w.replace('/', ' ')
			new_sent = new_sent.replace(w, replacement)
	words = nltk.word_tokenize(new_sent)
	tagged = nltk.pos_tag(words)
	for tuple_ in tagged:
		if (tuple_[1] == 'NNP'):
			name_repl = tuple_[0].replace(tuple_[0], '')
			new_sent = new_sent.replace(tuple_[0], name_repl)
	return str(new_sent)


def parse_xml():
	"""
	Reads the XML files, cleans them up appropriately to de-identify them, 
	and outputs the results as plain text in a separate directory
	"""
	for x in os.listdir("./XML/"):
		if str(x) == '.DS_Store':
			print("SKIP")
			continue
		filename = x[:len(x)-5]
		list_of_words = []
		tree = ET.parse('./XML/' + str(x))
		root = tree.getroot()
		for child in root: #head
			for toddler in child: #meta title div
				if (toddler.tag[len(toddler.tag)-3:] == 'div'):
					i = 0
					for baby in toddler: #p li
						if (i < 4):
							i = i + 1
							continue
						if ('well name' in str(baby.text).lower()):
							continue
						fn = False
						for token in filename.split():
							if (token.lower() in str(baby.text).lower()):
								fn = True
						if (fn):
							continue
						list_temp = []
						names_removed = remove_names(str(baby.text))
						list_temp.append(names_removed.split())
						for tmp in list_temp: #for each "list"
							for temp in tmp:
								numbers = sum(c.isdigit() for c in temp)
								if (numbers < 6):
									list_of_words.append(temp)
								else:
									continue
		print("========================================================================================================")
		with open('./XML_Stage2/' + filename, 'w') as output:
			for sent in list_of_words:
				output.write(str(sent) + ' ')


parse_xml()