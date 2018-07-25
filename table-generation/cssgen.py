#! /usr/bin/python

import itertools
import json
import csv
import os
import sys

class CSSGen(object):
    def __init__(self):
        self.folder = './CSS_NEW/'
        self.parameters = {}
        self.json_arr = []
        self.num_of_tables = 436700160
        self.num_of_files = 0

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)


    def build_json(self, index, lth, lty, lclr):
        """
        Builds a label JSON file that contains all information on the parameters
        :param index the index of the table
        :param lth line thickness
        :lty line type
        :lclr line colour
        :return the JSON data
        """
        data = {}
        data[str(index)] = [{
            'line_thickness': str(lth),
            'line_type': str(lty),
            'line_color': str(lclr)
        }]
        return data

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

    def write_border(self, line, thickness, style, clr):
        filename = os.path.join(self.folder, str(len(self.json_arr)) + ".css")
        fw = open(filename, "w+")
        fw.write(line)
        fw.close()

        json_dict = {
          "border_thickness": thickness,
          "border_type": style,
          "border_color": clr
        }
        self.json_arr.append(json_dict)

    def borders(self):
        """
        Generates CSS for table borders
        """
        cn = 0
        #css_param_counter = 0
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
                            self.write_border(line, thickness[th], style_css, clr_css)

                        total_clr_count += clr_count

                total_style_count += style_count

            total_thickness_count += thickness_count

    def write_to_arr(self, num_of_params, pars, nums, param_name, css_arr):
        """
        Writes the CSS
        :param num_of_params total number of parameters (e.g. 13 font families)
        :param pars names of the parameters
        :param nums queantities for each parameter
        :param param_name name of the parameter family
        :css_arr array of CSS elements
        :return css_arr updated array of CSS elements
        """
        counter = 0
        for a in range(num_of_params):
            for i in range(nums[a]):
                if counter >= self.num_of_files:
                    continue
                css_arr[counter] += "\n\t" + param_name + ": " + pars[a] + ";"
                print("# " + str(counter))
                self.json_arr[counter][param_name] = pars[a]
                counter += 1
        return css_arr


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
        list_of_nums = []
        list_of_pars = []
        for p in pars: #font-family
            l_n = []
            l_p = []
            for par in self.parameters: #font-family
                for pr in self.parameters[par]: # Georgia,serif
                    if par != p:
                        continue
                    if par == p:
                        pnum = int(self.num_of_tables * float(self.parameters[par][pr]))
                        if pr == 'no':
                            l_p.append('')
                            l_n.append(pnum)
                        else:
                            l_p.append(pr)
                            l_n.append(pnum)
            list_of_pars.append(l_p)
            list_of_nums.append(l_n)
        css_arr = []

        for filename in os.listdir(self.folder):
            css_arr.append("p {")
            self.num_of_files += 1
        for el in range(len(pars)):
            write_to_arr(num_of_pars[el], list_of_pars[el], list_of_nums[el], pars[el], css_arr)
        for c in range(len(css_arr)):
            css_arr[c] += "\n}\n"
        for el in range(len(css_arr)):
            print(css_arr[el])
            filename = os.path.join(self.folder, str(el) + ".css")
            with open(filename, "a") as f:
                f.write(css_arr[el])



    def fonts(self):
        """
        Font-specific CSS geenration
        """
        nums_ff = []
        nums_fst = []
        nums_fw = []
        nums_fv = []
        nums_fsz = []
        ff = []
        fst = []
        fw = []
        fv = []
        fsz = []
        for par in self.parameters:
            if 'border ' in par:
                continue
            if 'font-' in par:
                for pr in self.parameters[par]:
                    inum = int(self.num_of_tables * float(self.parameters[par][pr]))
                    if par == 'font-family':
                        nums_ff.append(inum)
                        ff.append(pr)
                    elif par == 'font-style':
                        nums_fst.append(inum)
                        fst.append(pr)
                    elif par == 'font-weight':
                        nums_fw.append(inum)
                        fw.append(pr)
                    elif par == 'font-variant':
                        nums_fv.append(inum)
                        fv.append(pr)
                    elif par == 'font-size':
                        nums_fsz.append(inum)
                        fsz.append(pr)
        print(nums_ff)

        css_arr = []

        for filename in os.listdir(self.folder):
            css_arr.append("p {")
            self.num_of_files += 1

        write_to_arr(13, ff, nums_ff, "font-family", css_arr)
        write_to_arr(3, fst, nums_fst, "font-style", css_arr)
        write_to_arr(2, fw, nums_fw, "font-weight", css_arr)
        write_to_arr(2, fv, nums_fv, "font-variant", css_arr)
        write_to_arr(10, fsz, nums_fsz, "font-size", css_arr)
        for c in range(len(css_arr)):
            css_arr[c] += "\n}\n"
        for el in range(len(css_arr)):
            print(css_arr[el])
            filename = os.path.join(this.folder,  str(el) + ".css")
            with open(filename, "a") as f:
                f.write(css_arr[el])

    def jsonDump(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for f in range(self.num_of_files):
            filename = os.path.join(folder, str(f) + '_str.json')
            with open(filename, 'w') as outfile:
                json.dump(self.json_arr[f], outfile)


def main(argv=sys.argv):
    """
    Generates the entire CSS and JSON label files
    """

    css = CSSGen()
    css.load_params()

    css.borders()


    css.general_table_css_generator("p", ["font-family", "font-style", "font-weight", "font-variant", "font-size"], "font-", [13, 3, 2, 2, 10])
    css.general_table_css_generator("table", ["width", "border-collapse"], "", [3, 2])
    css.general_table_css_generator("td", ["height", "text-align", "vertical-align"], "", [9, 3, 3])

    css.jsonDump('./JSON_NEW/');

if __name__ == "__main__":
    main()
