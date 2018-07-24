#! /usr/bin/python

import itertools
import json
import csv
import os

parameters = {}
# {'font-family': {'Georgia, serif': '0.02', '"Palatino Linotype", "Book Antiqua", Palatino, serif': '0.02', '"Times New Roman", Times, serif': '0.8', 'Arial, Helvetica, sans-serif': '0.06', '"Arial Black", Gadget, sans-serif': '0.05', '"Comic Sans MS", cursive, sans-serif': '0', 'Impact, Charcoal, sans-serif': '0', '"Lucida Sans Unicode", "Lucida Grande", sans-serif': '0', 'Tahoma, Geneva, sans-serif': '0.01', '"Trebuchet MS", Helvetica, sans-serif': '0', 'Verdana, Geneva, sans-serif': '0.01', '"Courier New", Courier, monospace': '0.03', '"Lucida Console", Monaco, monospace': '0'}, 'font-style': {'normal': '0.9', 'italic': '0.08', 'oblique': '0.02'}, 'font-weight': {'normal': '0.9', 'bold': '0.1'}, 'font-variant': {'normal': '0.85', 'small-caps': '0.15'}, 'font-size': {'8px': '0.1', '10px': '0.3', '12px': '0.2', '14px': '0.1', '16px': '0.1', '18px': '0.1', '20px': '0.1', '22px': '0', '24px': '0', '26px': '0'}, 'width': {'100%': '0.7', '75%': '0.2', '50%': '0.1'}, 'height': {'100px': '0.05', '500px': '0.15', '1000px': '0.2', '1500px': '0.2', '2000px': '0.3', '2500px': '0.1', '3000px': '0', '4000px': '0', '5000px': '0'}, 'padding': {'5px': '0.6', '10px': '0.2', '15px': '0.1', '20px': '0.1'}, 'border-collapses': {'yes': '0.3', 'no': '0.7'}, 'border thickness': {'1px': '0.7', '2px': '0.2', '3px': '0.1', '4px': '0'}, 'border type': {'solid': '0.95', 'dotted': '0.05'}, 'border color': {'black': '0.9', 'blue': '0.1', 'red': '0'}, 'text-align': {'left': '0.4', 'right': '0.4', 'center': '0.2'}, 'vertical-align': {'top': '0.4', 'bottom': '0.4', 'middle': '0.2'}, 'character distributions': {'words': '0.5', 'numbers': '0.4', 'symbols': '0.1'}}
num_of_tables = 436700160
num_of_files = 0
json_arr = []


def build_json(index, lth, lty, lclr):
    """
    Builds a label JSON file that contains all information on the parameters
    :param index the index of the table
    :param lth line thickness
    :lty line type
    :lclr line colour
    :return the JSON data
    """
    data = {}
    data[str(index)] = []
    data[str(index)].append({
        'line_thickness': str(lth),
        'line_type': str(lty),
        'line_color': str(lclr)
    })
    return data

def load_params(): 
    """
    Loads the csv files and builds a nested dictionary to represent all parameters and their percentages/probabilities
    """
    map_of_probs = {}
    main_keys = []
    counter_1 = 0
    counter_2 = 1
    with open('hyperparams.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        k_row = []
        v_row = []
        for row in readCSV:
            counter_1 = counter_1 + 1
            if counter_1 > 2:
                continue
            if counter_1 == 1:
                main_key = row[0]
                parameters[main_key] = {}
                k_row = row
                k_row.remove(k_row[0])
            elif counter_1 == 2:
                v_row = row
                v_row.remove(v_row[0])
        for key, value in zip(k_row, v_row):
            parameters[main_key][key] = value
    with open('./hyperparams.csv') as f:
        for line_keys, line_values in itertools.zip_longest(*[f]*2):
            if counter_2 == 1:
                counter_2 = counter_2 + 1
                continue
            k_row = line_keys.split(',')
            v_row = line_values.split(',')
            main_key = k_row[0]
            parameters[main_key] = {}
            k_row.remove(k_row[0])
            v_row.remove(v_row[0])
            for key, value in zip(k_row, v_row):
                if (len(key) <= 0 or key == '\n'):
                    continue
                parameters[main_key][key] = value
            counter_2 = counter_2 + 1


# def param_distr(numbers):


def borders():
    """
    Generates CSS for table borders
    """
    cn = 0
    border = False
    css_param_counter = 0 
    numbers_th = []
    numbers_st = []
    numbers_cl = []
    thickness = []
    style = []
    clr = []
    for par in parameters:
        css = ''
        css_param_counter += 1
        if css_param_counter == 16:
            continue
        if 'border ' in par:
            border = True
            for pr in parameters[par]:
                cn += 1
                if cn <= 4:
                    thickness.append(pr)
                    numbers_th.append(int(num_of_tables * float(parameters[par][pr])))
                if cn > 4 and cn <= 6:
                    style.append(pr)
                    numbers_st.append(int(num_of_tables * float(parameters[par][pr])))
                if cn > 6 and cn <= 9:
                    clr.append(pr)
                    numbers_cl.append(int(num_of_tables * float(parameters[par][pr])))
        else:
            css += str(par)
        css += '}\n'
    css_arr = []
    tmp_json_ar1 = []
    tmp_json_ar2 = []
    tmp_json_ar3 = []
    for th in range(4):
        for i in range(numbers_th[th]):
            css_arr.append("table, th, td, tr {\n\tborder: " + thickness[th] + " ")
            tmp_json_ar1.append(thickness[th])
    j = 0
    for s in range(2):
        while j < numbers_st[s]:
            css_arr[j] += style[s] + " "
            tmp_json_ar2.append(style[s])
            j += 1
    k = 0
    for c in range(3):
        while k < numbers_cl[c]:
            css_arr[k] += clr[c] + ";"
            tmp_json_ar3.append(clr[c])
            k += 1
    for el in range(len(css_arr)):
        css_arr[el] += "\n}\n"
    filecount = 0
    for v in css_arr:
        filename = str(filecount) + ".css"
        fw = open("./CSS_test/"+str(filename), "w+")
        fw.write(str(v))
        filecount += 1
        fw.close()
    global json_arr
    for j in range(filecount):
        json_dict = {}
        json_dict["border_thickness"] = tmp_json_ar1[j]
        json_dict["border_type"] = tmp_json_ar2[j]
        json_dict["border_color"] = tmp_json_ar3[j]
        json_arr.append(json_dict)


def write_to_arr(num_of_params, pars, nums, param_name, css_arr):
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
    global json_arr
    for a in range(num_of_params): 
        for i in range(nums[a]): 
            if counter >= num_of_files:
                continue
            css_arr[counter] += "\n\t" + param_name + ": " + pars[a] + ";"
            print("# " + str(counter))
            json_arr[counter][param_name] = pars[a]
            counter += 1
    return css_arr


def general_table_css_generator(tag, pars, section, num_of_pars):
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
        for par in parameters: #font-family
            for pr in parameters[par]: # Georgia,serif
                if par != p:
                    continue
                if par == p:
                    if pr == 'no':
                        l_p.append('')
                        l_n.append(int(num_of_tables * float(parameters[par][pr])))
                    else:
                        l_p.append(pr)
                        l_n.append(int(num_of_tables * float(parameters[par][pr])))
        list_of_pars.append(l_p)
        list_of_nums.append(l_n)
    css_arr = []
    path = './CSS_test/'
    global num_of_files
    for filename in os.listdir(path):
        css_arr.append("p {")
        num_of_files += 1
    for el in range(len(pars)):
        write_to_arr(num_of_pars[el], list_of_pars[el], list_of_nums[el], pars[el], css_arr)
    for c in range(len(css_arr)):
        css_arr[c] += "\n}\n"
    for el in range(len(css_arr)):
        print(css_arr[el])
        with open("./CSS_test/" + str(el) + ".css", "a") as f:
            f.write(css_arr[el])



def fonts():
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
    for par in parameters:
        if 'border ' in par:
            continue
        if 'font-' in par:
            for pr in parameters[par]:
                if par == 'font-family':
                    nums_ff.append(int(num_of_tables * float(parameters[par][pr])))
                    ff.append(pr)
                if par == 'font-style':
                    nums_fst.append(int(num_of_tables * float(parameters[par][pr])))
                    fst.append(pr)
                if par == 'font-weight':
                    nums_fw.append(int(num_of_tables * float(parameters[par][pr])))
                    fw.append(pr)
                if par == 'font-variant':
                    nums_fv.append(int(num_of_tables * float(parameters[par][pr])))
                    fv.append(pr)
                if par == 'font-size':
                    nums_fsz.append(int(num_of_tables * float(parameters[par][pr])))
                    fsz.append(pr)
    print(nums_ff)
    css_arr = []
    path = './CSS_test/'
    global num_of_files
    for filename in os.listdir(path):
        css_arr.append("p {")
        num_of_files += 1
    write_to_arr(13, ff, nums_ff, "font-family", css_arr)
    write_to_arr(3, fst, nums_fst, "font-style", css_arr)
    write_to_arr(2, fw, nums_fw, "font-weight", css_arr)
    write_to_arr(2, fv, nums_fv, "font-variant", css_arr)
    write_to_arr(10, fsz, nums_fsz, "font-size", css_arr)
    for c in range(len(css_arr)):
        css_arr[c] += "\n}\n"
    for el in range(len(css_arr)):
        print(css_arr[el])
        with open("./CSS_test/" + str(el) + ".css", "a") as f:
            f.write(css_arr[el])



def main():
    """
    Generates the entire CSS and JSON label files
    """
    load_params()
    global json_arr
    borders()
    general_table_css_generator("p", ["font-family", "font-style", "font-weight", "font-variant", "font-size"], "font-", [13, 3, 2, 2, 10])
    general_table_css_generator("table", ["width", "border-collapse"], "", [3, 2])
    general_table_css_generator("td", ["height", "text-align", "vertical-align"], "", [9, 3, 3])
    table_str_json = build_json(j, lth, lty, lclr)

    global num_of_files
    for f in range(num_of_files):
        with open('./JSON/' + str(f) + '_str.json', 'w') as outfile:
            json.dump(json_arr[f], outfile)


main()

