#! /usr/bin/python

import os
import csv

params = ['font-family', 'font-style', 'font-weight', 'font-variant', 'font-size', 'width', 'height', 
'padding', 'border-collapses', 'border thickness', 'border type', 'border color', 'border-bottom', 'text-align', 
'vertical-align', 'character distribution']

font_families = ['Georgia, serif', '"Palatino Linotype", "Book Antiqua", Palatino, serif', '"Times New Roman", Times, serif', 
'Arial, Helvetica, sans-serif', '"Arial Black", Gadget, sans-serif', '"Comic Sans MS", cursive, sans-serif', 'Impact, Charcoal, sans-serif',
'"Lucida Sans Unicode", "Lucida Grande", sans-serif', 'Tahoma, Geneva, sans-serif', '"Trebuchet MS", Helvetica, sans-serif',
'Verdana, Geneva, sans-serif', '"Courier New", Courier, monospace', '"Lucida Console", Monaco, monospace']
font_styles = ['normal', 'italic', 'oblique']
font_weights = ['normal', 'bold']
font_variants = ['normal', 'small-caps']
font_size = ['8px', '10px', '12px', '14px', '16px', '18px', '20px', '22px', '24px', '26px']

table_width = ['100%', '75%', '50%']
table_height = ['100px', '500px', '1000px', '1500px', '2000px', '2500px', '3000px', '4000px', '5000px']
padding = ['5px', '10px', '15px', '20px']

border_collapse = ['collapse', 'no']
border_thickness = ['1px', '2px', '3px', '4px']
border_type = ['solid', 'dotted']
border_colour = ['black', 'blue', 'red']

text_horiz_align = ['left', 'right', 'center']
text_vertic_align = ['top', 'bottom', 'middle']
character_distr = ['words', 'numbers', 'symbols']

def write_to_file(data):
	with open('./hyperparams.csv', 'a') as fp:
		wr = csv.writer(fp, dialect='excel')
		wr.writerow(data)


def main():
	data = []
	all_params = [font_families, font_styles, font_weights, font_variants, font_size, table_width, table_height, padding, border_collapse,
	border_thickness, border_type, border_colour, text_horiz_align, text_vertic_align, character_distr]

	for i in range(len(params)):
		var = input("Please enter percentage (in decimal) for each " + str(params[i]) + " seperated by a comma (e.g. 0.7,0.3)\n" + str(all_params[i]) + "\n")
		list_of_probs = var.split(',')
		all_params[i].insert(0, params[i])
		list_of_probs.insert(0, '')
		write_to_file(all_params[i])
		write_to_file(list_of_probs)


main()
