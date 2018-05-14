

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
                                  DIRECTION_VERTICAL,DIRECTION_HORIZONTAL)

# import sys
# from importlib import reload
# reload(sys)  # Reload does the trick!
# sys.setdefaultencoding('UTF8')

# %% Some constants
DATAPATH = '../../docs/'
OUTPUTPATH = '../../docs/'
INPUT_XML = 'demo3.pdf.xml'
N_ROW_BORDERS = 40
MIN_ROW_WIDTH=20
N_COL_BORDERS = 17
MIN_COL_WIDTH = 60  # <- very important! minimum width of a column in pixels, measured in the scanned pages


# %% Some helper functions
def save_image_w_lines(iproc_obj, imgfilebasename, orig_img_as_background):
    file_suffix = 'lines-orig' if orig_img_as_background else 'lines'

    img_lines = iproc_obj.draw_lines(orig_img_as_background=orig_img_as_background)
    img_lines_file = os.path.join(OUTPUTPATH, '%s-%s.png' % (imgfilebasename, file_suffix))

    print("> saving image with detected lines to '%s'" % img_lines_file)
    cv2.imwrite(img_lines_file, img_lines)


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
    DATAPATH='../../docs/'
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

    # detect the lines
    lines_hough = iproc_obj.detect_lines(canny_low_thresh=50, canny_high_thresh=150, canny_kernel_size=3,
                                         hough_rho_res=1,
                                         hough_theta_res=np.pi / 500,
                                         hough_votes_thresh=int(round(0.3 * iproc_obj.img_w)))
    print("> found %d lines" % len(lines_hough))

    save_image_w_lines(iproc_obj, imgfilebasename, True)
    save_image_w_lines(iproc_obj, imgfilebasename, False)




    horizontal_clusters = iproc_obj.find_clusters(imgproc.DIRECTION_HORIZONTAL, find_clusters_1d_break_dist,
                                                remove_empty_cluster_sections_use_texts=p['texts'],
                                                # use this page's textboxes
                                                remove_empty_cluster_sections_n_texts_ratio=0.1,  # 10% rule
                                                remove_empty_cluster_sections_scaling=page_scaling_y,
                                                # the positions are in "scanned image space" -> we scale them to "text box space"
                                                dist_thresh=MIN_ROW_WIDTH / 2)
    print("> found %d clusters" % len(horizontal_clusters))

    # draw the clusters
    img_h_clusters = iproc_obj.draw_line_clusters(imgproc.DIRECTION_HORIZONTAL, horizontal_clusters)
    save_img_file = os.path.join(OUTPUTPATH, '%s-horizontal-clusters.png' % imgfilebasename)
    print("> saving image with detected horizontal clusters to '%s'" % save_img_file)
    cv2.imwrite(save_img_file, img_h_clusters)

    horizontal_lines_clusters[p_num] = horizontal_clusters



    # %% Get column positions as adjusted vertical line clusters
    print("calculating column positions for all pages...")

    pages_image_scaling_x = {p_num: sx for p_num, (sx, _) in pages_image_scaling.items()}


    # %% Get column positions as adjusted vertical line clusters
    print("calculating row positions for all pages...")

    pages_image_scaling_x = {p_num: sx for p_num, (sx, _) in pages_image_scaling.items()}
    pages_image_scaling_y = {p_num: sy for p_num, (_, sy) in pages_image_scaling.items()}

    row_positions = get_adjusted_cluster_centers(horizontal_lines_clusters, N_ROW_BORDERS,
                                                 find_center_clusters_method=find_clusters_1d_break_dist,
                                                 dist_thresh=MIN_ROW_WIDTH / 2,
                                                 image_scaling=pages_image_scaling_y)  # the positions are in "scanned


    row_positions_noscale = get_adjusted_cluster_centers(horizontal_lines_clusters, N_ROW_BORDERS,
                                                 find_center_clusters_method=find_clusters_1d_break_dist,
                                                 dist_thresh=MIN_ROW_WIDTH / 2,
                                                 image_scaling={1:1})  # the positions are in "scanned


    baseimg = iproc_obj._baseimg_for_drawing(True)
    iproc_obj.draw_lines_in_dir(baseimg, DIRECTION_HORIZONTAL, list(row_positions_noscale[p_num]), (225, 0, 0))
    save_img_file = os.path.join(OUTPUTPATH, '%s-horizontal-lines.png' % imgfilebasename)
    cv2.imwrite(save_img_file,baseimg)

    row_positions_noscale=row_positions_noscale[p_num]


    # divide the image into segments
    print ('Processing segments.....')

    prev_pos=0
    for i, pos in enumerate(row_positions_noscale):

        if i==0:
            prev_pos=pos
            continue


        img_segment=iproc_obj.input_img[int(prev_pos):int(pos),0:iproc_obj.img_w]
        iproc_obj_segment = imgproc.ImageProc(imgfile,input_img=img_segment)
        lines_hough = iproc_obj_segment.detect_lines_inv(canny_low_thresh=50, canny_high_thresh=150, canny_kernel_size=3,
                                                 hough_rho_res=1,
                                                 hough_theta_res=np.pi / 500,
                                                 hough_votes_thresh=int(round(0.99 * iproc_obj_segment.img_h)))
        print("> found %d lines" % len(lines_hough))



        save_image_w_lines(iproc_obj_segment, imgfilebasename+"_"+str(i), True)
        save_image_w_lines(iproc_obj_segment, imgfilebasename+"_"+str(i), False)

        # cluster the detected *vertical* lines using find_clusters_1d_break_dist as simple clustering function
        # (break on distance MIN_COL_WIDTH/2)
        # additionally, remove all cluster sections that are considered empty
        # a cluster is considered empty when the number of text boxes in it is below 10% of the median number of text boxes
        # per cluster section
        vertical_clusters = iproc_obj_segment.find_clusters(imgproc.DIRECTION_VERTICAL, find_clusters_1d_break_dist,
                                                    remove_empty_cluster_sections_use_texts=p['texts'],
                                                    # use this page's textboxes
                                                    remove_empty_cluster_sections_n_texts_ratio=0.1,  # 10% rule
                                                    remove_empty_cluster_sections_scaling=page_scaling_x,
                                                    # the positions are in "scanned image space" -> we scale them to "text box space"
                                                    dist_thresh=MIN_COL_WIDTH / 2)
        print("> found %d segment clusters" % len(vertical_clusters))

        img_vert_lines_file = iproc_obj_segment.draw_lines_filtered_straight(imgproc.DIRECTION_VERTICAL)
        save_img_file = os.path.join(OUTPUTPATH, '%s-vertical_filetred_lines.png' % imgfilebasename)
        cv2.imwrite(save_img_file, img_vert_lines_file)
        # draw the clusters
        img_w_clusters = iproc_obj_segment.draw_line_clusters(imgproc.DIRECTION_VERTICAL, vertical_clusters)

        save_img_file = os.path.join(OUTPUTPATH, imgfilebasename+'-vertical-clusters'+"_"+str(i)+'.png')
        cv2.imwrite(save_img_file, img_w_clusters)
        vertical_lines_clusters[p_num] = vertical_clusters

        col_positions_noscale = get_adjusted_cluster_centers(vertical_lines_clusters, N_COL_BORDERS,
                                                             find_center_clusters_method=find_clusters_1d_break_dist,
                                                             dist_thresh=MIN_COL_WIDTH / 2,
                                                             image_scaling={1: 1})  # the positions are in "scanned

        baseimg = iproc_obj_segment._baseimg_for_drawing(True)
        iproc_obj.draw_lines_in_dir(baseimg, DIRECTION_VERTICAL, list(col_positions_noscale[p_num]), (225, 0, 0))
        save_img_file = os.path.join(OUTPUTPATH, imgfilebasename+'-vertical-clusters'+"_"+str(i)+'.png')
        cv2.imwrite(save_img_file, baseimg)


        # vertical_lines_clusters[p_num] = vertical_clusters


        # iterate through each segment and apply column detection:


        prev_pos = pos



