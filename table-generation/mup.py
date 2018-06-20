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

items = ("Item one", "Item two", "Item three", "Item four")
paras = ("This was a fantastic list.", "And now for something completely different.")
images = ("thumb1.jpg", "thumb2.jpg", "more.jpg", "more2.jpg")

start = time.time()

def table_to_pdf(name):
    pdfkit.from_file('./HTML/' + name + ".html", './PDF/' + name + ".pdf")

def pdf_to_png(name):
    size = 7016, 4961
    with Image(filename='./PDF/' + name + '.pdf') as img:
        #print('pages = ', len(img.sequence))
        with img.convert('png') as converted:
            converted.save(filename='./PNG/' + name+ '.png')

def pdf_to_jpg(filepdf, name):
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
    #return path

def bounding_box(image_file):
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
    #print(box)

def write_to_csv(data_arr):
    f = open('bounding_boxes.csv','w')
    for i in range(len(data_arr)):
        temp = data_arr[i]
        for j in range(3):
            f.write(str(temp[j]) + ', ')
        f.write(str(temp[3]) + '\n')
    f.close()


bounding_boxes_ = []
for x in os.listdir("./CSS"):
    page = mp.page()
    page.init(title="HTML Generator",
              css=('../CSS/' + str(x)))
    page.table()

    for i in range(10):
        page.tr()
        for j in range(5):
            page.td(j)
            page.td.close()
        page.tr.close()

    page.table.close()

    file_ = os.path.splitext(x)[0]
    filename = file_ + ".html"
    fw = open("./HTML/" + filename, "w+")
    fw.write(str(page))
    fw.close()

    bounding_boxes_.append(pdf_to_jpg('./PDF/' + file_ + '.pdf', file_))

write_to_csv(bounding_boxes_)

end = time.time()
print(end - start)
