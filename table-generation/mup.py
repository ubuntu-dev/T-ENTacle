#! /usr/bin/python

from __future__ import print_function
import MarkupPy.markup as mp
import webbrowser as web
import os
import time
from wand.image import Image
from PIL import Image as Img
import numpy as np
import pdfkit
import tinycss
import uuid
import glob
import sys
import mxnet as mx
import cv2
import csv
import random

items = ("Item one", "Item two", "Item three", "Item four")
paras = ("This was a fantastic list.", "And now for something completely different.")
images = ("thumb1.jpg", "thumb2.jpg", "more.jpg", "more2.jpg")

start = time.time()

def table_to_pdf(name):
    """
    Converts an HTML table to a PDF file
    :param name file name
    """
    pdfkit.from_file('./HTML/' + name + ".html", './PDF/' + name + ".pdf")


def pdf_to_png(name):
    """
    Converts the PDF of a table to a PNG image
    :param name file name
    """
    size = 7016, 4961
    with Image(filename='./PDF/' + name + '.pdf') as img:
        with img.convert('png') as converted:
            converted.save(filename='./PNG/' + name+ '.png')


def pdf_to_jpg(filepdf, name):
    """
    Converts the PDF of a table to a JPG image
    :filepdf the PDF file
    :param name file name
    :return the bounding box of the table in the JPG
    """
    uuids = str(uuid.uuid4().fields[-1])[:5]
    with Image(filename=filepdf, resolution=500) as img:
        img.compression_quality = 80
        img.save(filename='TMP/temp_' + name + '_%s.jpg' % uuids)
        list_im = glob.glob('TMP/temp_' + name + '_%s.jpg' % uuids)
        list_im.sort()
        imgs = [Img.open(i) for i in list_im]
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1] #combine images
        imgs_c = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs)) #vertical
        imgs_c = Img.fromarray(imgs_c)
        path = './JPG/' + name + '_%s.jpg' % uuids
        imgs_c.save(path)
        bounding_box(path)
        for i in list_im:
            os.remove(i)
    return bounding_box(path)


def bounding_box(image_file):
    """
    Detects the bounding box of a table in an image using Open CV
    and reports its coordinates
    :param image_file the image to be analysed
    """
    im = cv2.imread(image_file)
    im[im == 255] = 1
    im[im == 0] = 255
    im[im == 1] = 0
    im2 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(im2,127,255,0)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt) #The function calculates and returns the minimal up-right bounding rectangle for the specified point set.
    x1 = x + (w-1)
    y1 = y + (h-1)

    out = [x, y, x1, y1]
    print(out)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)

    return out


def write_to_csv(data_arr):
    """
    Writes the bounding box data to a CSV file
    :param data_arr the data to be written to a CSV
    """
    f = open('bounding_boxes.csv','w')
    for i in range(len(data_arr)):
        temp = data_arr[i]
        for j in range(3):
            f.write(str(temp[j]) + ', ')
        f.write(str(temp[3]) + '\n')
    f.close()


def read_words(words_file):
    """
    Reads a text file and puts its words into a list
    :param words_file text file to be read
    :return list of words read from a file
    """
    return [word for line in open(words_file, 'r') for word in line.split()]


def grouped(list_of_words, n):
    return zip((*[iter(list_of_words)]*n))


def generate_html():
    bounding_boxes_ = []
    for x in os.listdir("./CSS"):
        page = mp.page()
        page.init(title="HTML Generator",
                  css=('../CSS/' + str(x)))
        page.table()

        list_of_words = read_words('../pdf-parser/text_for_tables.txt')

        for i in range(10): #rows
            page.tr()
            for j in range(5): #columns
                num = random.randint(1, 21)
                temp = ''
                for r in range(num):
                    temp += random.choice(list_of_words) + ' '
                page.td(temp)
                page.td.close()
            page.tr.close()

        page.table.close()

        file_ = os.path.splitext(x)[0]
        filename = file_ + ".html"
        fw = open("./HTML/" + filename, "w+")
        fw.write(str(page))
        fw.close()

        table_to_pdf(file_)
        pdf_to_jpg('./PDF/' + file_ + '.pdf', file_)

        #bounding_boxes_.append(pdf_to_jpg('./PDF/' + file_ + '.pdf', file_))

    #write_to_csv(bounding_boxes_)

end = time.time()
print(end - start)

generate_html()
# test = [1,2,3,4,5]
# for x, y in grouped(test, 2):
#    print(str(x) + str(y))
