

import os
import re
from math import radians, degrees

import numpy as np
import pandas as pd
import cv2

import imageproc as imgproc
from pdftabextract.geom import pt
from pdftabextract.textboxes import (border_positions_from_texts, split_texts_by_positions, join_texts,
                                     rotate_textboxes, deskew_textboxes)
from clustering import (find_clusters_1d_break_dist,
                                      calc_cluster_centers_1d,
                                      zip_clusters_and_values,
                                      get_adjusted_cluster_centers)

from pdftabextract.extract import make_grid_from_positions, fit_texts_into_grid, datatable_to_dataframe
from pdftabextract.common import (read_xml, parse_pages, save_page_grids, all_a_in_b,
                                  ROTATION, SKEW_X, SKEW_Y,
                                  DIRECTION_VERTICAL)
# import sys
# from importlib import reload
# reload(sys)  # Reload does the trick!
# sys.setdefaultencoding('UTF8')

# %% Some constants
DATAPATH = 'docs/'
OUTPUTPATH = 'docs/'
INPUT_XML = 'demo3.pdf.xml'


# %% Read the XML

# Load the XML that was generated with pdftohtml
xmltree, xmlroot = read_xml(os.path.join(DATAPATH, INPUT_XML))

# parse it and generate a dict of pages
pages = parse_pages(xmlroot)

# %% Detect clusters of vertical lines using the image processing module and rotate back or deskew pages

vertical_lines_clusters = {}
horizontal_lines_clusters = {}
pages_image_scaling = {}  # scaling of the scanned page image in relation to the OCR page dimensions for each page

for p_num, p in pages.items():
    # get the image file of the scanned page
    imgfilebasename='demo3.png'
    DATAPATH='docs/'
    imgfile = os.path.join(DATAPATH, imgfilebasename)

    print("page %d: detecting lines in image file '%s'..." % (p_num, imgfile))

    # create an image processing object with the scanned page
    iproc_obj = imgproc.ImageProc(imgfile)

    # calculate the scaling of the image file in relation to the text boxes coordinate system dimensions
    page_scaling_x = float(iproc_obj.img_w) / p['width']
    page_scaling_y = float(iproc_obj.img_h) / p['height']
    # print 'x-scale',iproc_obj.img_w,p['width']
    pages_image_scaling[p_num] = (page_scaling_x,  # scaling in X-direction
                                  page_scaling_y)  # scaling in Y-direction

    lines_hough = iproc_obj.detect_lines(canny_low_thresh=50, canny_high_thresh=150, canny_kernel_size=3,
                                         hough_rho_res=1,
                                         hough_theta_res=np.pi / 500,
                                         hough_votes_thresh=int(round(0.3 * iproc_obj.img_w)))

    cv2.imwrite("docs/image_edges.png", iproc_obj.edges)

    h = iproc_obj.edges.shape[0]
    w = iproc_obj.edges.shape[1]
    print(h,w)
    thresh_vert=50
    thresh_hori = 50
    vertical_lines=[]
    horizontal_lines=[]
    baseimg = iproc_obj._baseimg_for_drawing(True)
    horizontal_img = iproc_obj._baseimg_for_drawing(True)
    vertical_img = iproc_obj._baseimg_for_drawing(True)

    for x in range(0, w):
        buffer=[]
        previous_pixel=-1
        for y in range(0, h):
            # print(iproc_obj.edges[y,x])
            if y==0:
                buffer.append(y)
            elif iproc_obj.edges[y,x]==previous_pixel:
                buffer.append(y)
            else:
                if len(buffer)>thresh_vert:
                    vertical_lines.append(((x,buffer[0]),(x,buffer[-1])))
                    cv2.line(horizontal_img, (x,buffer[0]),(x,buffer[-1]), (0, 0, 225), 1)

                buffer=[]
            previous_pixel=iproc_obj.edges[y,x]


    baseimg = cv2.addWeighted(baseimg, 0.8, horizontal_img, 0.2,0)
    # cv2.imwrite("docs/whoknows_image_vertical.png", baseimg)

    # baseimg = iproc_obj._baseimg_for_drawing(True)

    for y in range(0, h):
        buffer=[]
        previous_pixel=-1
        for x in range(0, w):
            # print(iproc_obj.edges[y,x])
            if x==0:
                buffer.append(x)
            elif iproc_obj.edges[y,x]==previous_pixel:
                buffer.append(x)
            else:
                if len(buffer)>thresh_hori:
                    horizontal_lines.append(((buffer[0],y),(buffer[-1],y)))
                    cv2.line(vertical_img, (buffer[0],y),(buffer[-1],y), (255, 255, 0), 1)

                buffer=[]
            previous_pixel=iproc_obj.edges[y,x]


    baseimg = cv2.addWeighted(baseimg, 0.8, vertical_img, 0.2,0)
    # cv2.imwrite("docs/whoknows_image_horizontal.png", baseimg)

    cv2.imwrite("docs/whoknows_image.png", baseimg)



