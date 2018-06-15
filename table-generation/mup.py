#! /usr/bin/python

import MarkupPy.markup as mp
import webbrowser as web
import os
import time

items = ("Item one", "Item two", "Item three", "Item four")
paras = ("This was a fantastic list.", "And now for something completely different.")
images = ("thumb1.jpg", "thumb2.jpg", "more.jpg", "more2.jpg")

start = time.time()

for x in os.listdir("D:\\5thSemester_Summer2018\\PyHTML\\CSS"):
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
    # page.img(src=images, width=100, height=80, alt="Thumbnails")

    file_ = os.path.splitext(x)[0]
    filename = file_ + ".html"

    fw = open("D:\\5thSemester_Summer2018\\PyHTML\\HTML\\"+filename, "w+")
    fw.write(str(page))
    fw.close()

    web.open_new("http://localhost:63342/PyHTML/HTML/"+filename+"?_ijt=uot62amf4ihap3vqcrj1s6mv3j")

end = time.time()
print(end - start)
