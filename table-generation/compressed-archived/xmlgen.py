#! /usr/bin/python

import csv
import os
import xml.etree.ElementTree as ET

image_dict = {}

def read_csv(filename): #./css/table_locations.csv
	"""
	Reads the csv file and yields the data (width, height, top, left)
	:param filename the csv file
	:return width, height, top, left
	"""
	with open(filename) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		for i, row in enumerate(readCSV):
			if i == 0:
				continue
			yield(int(row[0]), int(row[1]), int(row[2]), int(row[3]))


def read_image_files():
	"""
	Reads png files and builds a dictionary
	:return image dictionary
	"""
	temp = []
	for f in sorted(os.listdir('./png/')):
		if "-" not in f:
			image_dict[f[:-4]] = ".png"
		else:
			file = f.split("-")
			image_dict[file[0]] = []
	for fl in sorted(os.listdir('./png/')):
		if "-" not in fl:
			continue
		else:
			file = fl.split("-")
			l = list(image_dict[file[0]])
			l.append("-" + file[1])
			image_dict[file[0]] = l
	for el in image_dict:
		if image_dict[el][0] == '.':
			ls = list(image_dict[el])
			ls[0:4] = [''.join(ls[0:4])]
			image_dict[el] = ls
	return image_dict
		

def generate_xml(w, h, t, l, filenum):
	"""
	Generates the xml file
	:param w width
	:param h height
	:param t top
	:param l left
	:param filenum the file number
	"""
	xmax = l + w
	ymax = t + h
	root = ET.Element("annotation")
	ET.SubElement(root, "folder").text = "train"
	if image_dict[filenum] == ".png":
		ET.SubElement(root, "filename").text = str(filenum) + str(image_dict[filenum])
	else:
		filename = ET.SubElement(root,"filename")
		for i, ext in enumerate(image_dict[filenum]):
			ET.SubElement(filename, str(i)).text = str(filenum) + ext
	size = ET.SubElement(root,"size")
	ET.SubElement(size, "width").text = str(w)
	ET.SubElement(size, "height").text = str(h)
	bndbox = ET.SubElement(root, "bndbox")
	ET.SubElement(bndbox, "xmin").text = str(l)
	ET.SubElement(bndbox, "ymin").text = str(t)
	ET.SubElement(bndbox, "xmax").text = str(xmax)
	ET.SubElement(bndbox, "ymax").text = str(ymax)

	tree = ET.ElementTree(root)
	tree.write('./xml/' + str(filenum) + '.xml')


def main():
	"""
	Performs all required operations
	"""
	xml_data = read_csv('./css/table_locations.csv')
	images_dict = read_image_files()
	for data, file in zip(xml_data, images_dict):
		w, h, t, l = data
		generate_xml(w, h, t, l, file)


main()