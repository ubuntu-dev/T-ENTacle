#! /usr/bin/python

import csv
import os
import xml.etree.ElementTree as ET

def read_csv(filename): #./css/table_locations.csv
	with open(filename) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		for i, row in enumerate(readCSV):
			if i == 0:
				continue
			yield(row[0], row[1], row[2], row[3])


def read_image_files():
	image_dict = {}
	temp = []
	for i, f in enumerate(sorted(os.listdir('./png/'))):
		file = f.split("-")
		# print(file)
		if "-" not in f:
			image_dict[file[0][:-4]] = ".png"
		else:
			image_dict[file[0]] = []
	print()
	print()
	print()

	for i, f in enumerate(sorted(os.listdir('./png/'))):
		if "-" not in f:
			continue
		file = f.split("-")
		# temp.append(file[1])
		# print(file)
		# print(image_dict[file[0]])
		image_dict[file[0]].append("-" + file[1])

	# print(image_dict)
	# return image_dict
		



def generate_xml(w, h, t, l, count):
	root = ET.Element("annotation")
	# folder = ET.SubElement(root, "folder")
	ET.SubElement(root, "folder").text = "train"
	
	filename = ET.SubElement(root,"filename")
	size = ET.SubElement(root,"size")
	# width = ET.SubElement(size, "width")
	# height = ET.SubElement(size, "height")
	ET.SubElement(size, "width").text = w
	ET.SubElement(size, "height").text = h


	tree = ET.ElementTree(root)
	# ET.dump(tree)
	# tree.write('./xml/' + str(count) + '.xml')
	# mydata = ET.tostring(data)  
	# myfile = open("items2.xml", "w")  
	# myfile.write(mydata)


def main():
	xml_data = read_csv('./css/table_locations.csv')
	count = 0
	for data in xml_data:
		w, h, t, l = data
		generate_xml(w, h, t, l, count)
		count += 1



# main()
read_image_files()