#! /usr/bin/python

import itertools
import json
import csv
import os
import sys
from time import time

from queue import Queue
from threading import Thread

import tarfile
import io

class CSSGen(object):

    def __init__(self):
        self.parameters = {}
        self.num_of_tables = 436700160
        

    def load_params(self):
        """
        Loads the csv files and builds a nested dictionary to represent all parameters and their percentages/probabilities
        """
        map_of_probs = {}
        main_keys = []
        counter = 0

        with open('hyperparams.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            k_row = []
            v_row = []
            for row in readCSV:
                counter = counter + 1
                if counter > 2:
                    break
                if counter == 1:
                    main_key = row[0]
                    self.parameters[main_key] = {}
                    k_row = row
                    k_row.remove(k_row[0])
                elif counter == 2:
                    v_row = row
                    v_row.remove(v_row[0])
            for key, value in zip(k_row, v_row):
                self.parameters[main_key][key] = value

        with open('./hyperparams.csv') as f:
            counter = 0
            for line_keys, line_values in itertools.zip_longest(*[f]*2):
                if counter == 0:
                    counter = counter + 1
                    continue

                k_row = line_keys.split(',')
                v_row = line_values.split(',')
                main_key = k_row[0]
                self.parameters[main_key] = {}
                k_row.remove(k_row[0])
                v_row.remove(v_row[0])
                for key, value in zip(k_row, v_row):
                    if (len(key) <= 0 or key == '\n'):
                        continue
                    self.parameters[main_key][key] = value


    def borders(self):
        """
        Generates CSS for table borders
        """
        cn = 0
        numbers_th = []
        numbers_st = []
        numbers_cl = []
        thickness = []
        style = []
        clr = []

        item = self.parameters['border thickness']
        for pr in item:
            pnum = int(self.num_of_tables * float(item[pr]))
            thickness.append(pr)
            numbers_th.append(pnum)

        item = self.parameters['border color']
        for pr in item:
            pnum = int(self.num_of_tables * float(item[pr]))
            clr.append(pr)
            numbers_cl.append(pnum)

        item = self.parameters['border type']
        for pr in item:
            pnum = int(self.num_of_tables * float(item[pr]))
            style.append(pr)
            numbers_st.append(pnum)

        total_thickness_count = 0

        for th in range(len(numbers_th)):
            thickness_css = "table, th, td, tr {\n\tborder: " + thickness[th]
            thickness_count = numbers_th[th]

            total_style_count = 0
            for s in range(len(numbers_st)):
                style_css = style[s]
                style_count = numbers_st[s]

                start = max(total_thickness_count, total_style_count)
                end = min(total_thickness_count + thickness_count, total_style_count + style_count)

                if start <= end:
                    total_clr_count = 0
                    for c in range(len(numbers_cl)):
                        clr_count = numbers_cl[c]
                        clr_css = clr[c]
                        start = max(start, total_clr_count)
                        end = min(end, total_clr_count + clr_count)
                        line = thickness_css + " " + style_css + " " + clr_css + ";\n}\n"
                        while start <= end:
                            json_dict = {
                              "border_thickness": thickness[th],
                              "border_type": style_css,
                              "border_color": clr_css
                            }
                            yield (line, json_dict)

                        total_clr_count += clr_count

                total_style_count += style_count

            total_thickness_count += thickness_count


    def general_table_css_generator(self, tag, pars, section, num_of_pars):
        """
        Generates general CSS for tables
        :param tag the tag for the CSS element
        :param pars array of parameter names
        :param section the parameters section
        :param num_of_pars array of numbers of parameters
        """
        if section == 'border ':
            sys.exit("Border settings require a separate function!")

        for param_name, num_of_params in zip(pars, num_of_pars): #font-family
            if param_name in self.parameters:
                l_nums = []
                l_pars = []
                for pr in self.parameters[param_name]:
                    pnum = int(self.num_of_tables * float(self.parameters[param_name][pr]))
                    if pr == 'no':
                        l_pars.append('')
                        l_nums.append(pnum)
                    else:
                        l_pars.append(pr)
                        l_nums.append(pnum)

                for a in range(num_of_params):
                    line = tag + " {\n\t" + param_name + ": " + l_pars[a] + ";\n}\n"
                    csv_line = ''
                    dic = { }
                    dic[param_name] = l_pars[a]
                    if tag == 'table':
                        if param_name == 'width' or param_name == 'height' or param_name == 'top' or param_name == 'left':
                            sz = l_pars[a]
                            csv_line += str(sz[:-2]) + ','
                        csv_line = csv_line[:-1]
                    for i in range(l_nums[a]):
                        yield (line, dic, csv_line)


class CssJsonWriterWorker(Thread):

   def __init__(self, css_folder, json_folder, queue):
    Thread.__init__(self)
    self.css_folder = css_folder
    self.json_folder = json_folder
    self.queue = queue

   def run(self):
    while True:
        ind, css, dic = self.queue.get()

        filename = os.path.join(self.css_folder, ind + ".css")
        with open(filename, 'w') as file:
            file.write(css)

        filename = os.path.join(self.json_folder, ind + '_str.json')
        with open(filename, 'w') as outfile:
            json.dump(dic, outfile)

        self.queue.task_done()


def main(argv=sys.argv):
    """
    Generates the entire CSS and JSON label files
    """
    ts = time()
    css = CSSGen()
    css.load_params()

    borders = css.borders()
    paragraphs =  css.general_table_css_generator("p", ["font-family", "font-style", "font-weight", "font-variant", "font-size"], "font-", [13, 3, 2, 2, 10])
    tables = css.general_table_css_generator("table", ["width", "height", "position", "top", "left", "border-collapse"], "", [3, 2])
    tds = css.general_table_css_generator("td", ["padding", "text-align", "vertical-align"], "", [9, 3, 3])

    css_folder = './css'
    if not os.path.exists(css_folder):
        os.makedirs(css_folder)

    json_folder = './json'
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    # queue = Queue()
    # thread_count = 10
    # for x in range(thread_count):
    #     worker = CssJsonWriterWorker(css_folder, json_folder, queue)
    #     worker.daemon = True
    #     worker.start()

    ts = time()
    count = 0

    # css_tar = tarfile.open("css.tar", "w")
    css_tar = tarfile.open("css.tar.gz", "w:gz")
    json_tar = tarfile.open("json.tar.gz", "w:gz")

    csv_coordinates = open('./table_locations.csv', 'w')
    csv_coordinates.write('width,height,top,left\n')

    for border, paragraph, table, td in zip(borders, paragraphs, tables, tds):
        line1, dic1 = border
        line2, dic2, csv_line2 = paragraph
        line3, dic3, csv_line3 = table
        line4, dic4, csv_line4 = td

        name = str(count)
        css_content = line1 + line2 + line3 + line4

        css_bytes = io.BytesIO(css_content.encode('utf8'))
        css_info = tarfile.TarInfo(name= "css/" + name + ".css")
        css_info.size=len(css_content)
        css_tar.addfile(tarinfo = css_info, fileobj = css_bytes)

        dic = {**dic1, **dic2, **dic3, **dic4}
        json_content = json.dumps(dic)

        json_bytes = io.BytesIO(json_content.encode('utf8'))
        json_info = tarfile.TarInfo(name= "json/" + name + "_str.json")
        json_info.size=len(json_content)
        css_tar.addfile(tarinfo = json_info, fileobj = json_bytes)

        csv_coordinates.write(csv_line3 + '\n')


        # queue.put((str(count), line1 + line2 + line3 + line4, dic))

        # filename = os.path.join(css_folder, str(count) + ".css")
        # with open(filename, 'w') as file:
        #     file.write(line1)
        #     file.write(line2)
        #     file.write(line3)
        #     file.write(line4)

        # # dic = {**dic1, **dic2, **dic3, **dic4}
        # filename = os.path.join(json_folder, str(count) + '_str.json')
        # with open(filename, 'w') as outfile:
        #     json.dump(dic, outfile)

        count += 1

    css_tar.close()
    json_tar.close()
    csv_coordinates.close()
    #queue.join()
    print('Took {}'.format(time() - ts))


if __name__ == "__main__":
    main()