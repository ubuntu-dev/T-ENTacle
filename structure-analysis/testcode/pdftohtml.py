from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import HTMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')

def convert_pdf_to_html(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = HTMLConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = file(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0 # is for all
    caching = True
    pagenos=set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
        interpreter.process_page(page)
    fp.close()
    device.close()
    str = retstr.getvalue()
    retstr.close()
    return str



# print convert_pdf_to_html('/Users/asitangm/Desktop/pdfs 2/Anadarko/Banshee Well Reports.pdf')

import os
from pdftabextract.common import read_xml, parse_pages
from pdftabextract import imgproc

DATAPATH = 'docs/'
OUTPUTPATH = 'generated_output/'
INPUT_XML = 'demo.pdf.xml'
# Load the XML that was generated with pdftohtml
xmltree, xmlroot = read_xml(os.path.join(DATAPATH, INPUT_XML))

# parse it and generate a dict of pages
pages = parse_pages(xmlroot)
p=pages[1]

import numpy as np
from pdftabextract import imgproc

# get the image file of the scanned page
imgfile = os.path.join(DATAPATH, 'demo.pdf-1_1.png')
iproc_obj = imgproc.ImageProc(imgfile)
page_scaling_x = iproc_obj.img_w / p['width']   # scaling in X-direction
page_scaling_y = iproc_obj.img_h / p['height']  # scaling in Y-direction

# detect the lines
lines_hough = iproc_obj.detect_lines(canny_kernel_size=3, canny_low_thresh=50, canny_high_thresh=150,
                                     hough_rho_res=1,
                                     hough_theta_res=np.pi/500,
                                     hough_votes_thresh=int(round(1 * iproc_obj.img_h)))
print("> found %d lines" % len(lines_hough))