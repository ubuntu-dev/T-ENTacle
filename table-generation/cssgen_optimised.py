#! /usr/bin/python

import itertools
import json
import csv
import os
import sys

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
                              "border_thickness": thickness,
                              "border_type": style,
                              "border_color": clr
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
                    dic = { }
                    dic[param_name] = l_pars[a]
                    for i in range(l_nums[a]):
                        yield (line, dic)


def main(argv=sys.argv):
    """
    Generates the entire CSS and JSON label files
    """

    css = CSSGen()
    css.load_params()

    borders = css.borders()
    paragraphs =  css.general_table_css_generator("p", ["font-family", "font-style", "font-weight", "font-variant", "font-size"], "font-", [13, 3, 2, 2, 10])
    tables = css.general_table_css_generator("table", ["width", "border-collapse"], "", [3, 2])
    tds = css.general_table_css_generator("td", ["height", "text-align", "vertical-align"], "", [9, 3, 3])

    css_folder = './CSS_NEW_1'
    if not os.path.exists(css_folder):
        os.makedirs(css_folder)

    json_folder = './JSON_NEW_1'
    if not os.path.exists(json_folder):
            os.makedirs(json_folder)

    count = 0
    for border, paragraph, table, td in zip(borders, paragraphs, tables, tds):
        line1, dic1 = border
        line2, dic2 = paragraph
        line3, dic3 = table
        line4, dic4 = td

        filename = os.path.join(css_folder, str(count) + ".css")
        with open(filename, 'w') as file:
            file.write(line1)
            file.write(line2)
            file.write(line3)
            file.write(line4)

        dic = {**dic1, **dic2, **dic3, **dic4}
        filename = os.path.join(json_folder, str(count) + '_str.json')
        with open(filename, 'w') as outfile:
            json.dump(dic, outfile)

        count += 1

if __name__ == "__main__":
    main()