#! /usr/bin/python

import itertools
import json
import csv

parameters = {}

def build_json(index, lth, lty, lclr):
    """
    Builds a label JSON file that contains all information on the parameters
    :param index the index of the table
    :lth line thickness
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


def main():
    """
    Performs CSS and JSON generation
    """
    # line_type = ['dotted', 'solid']
    # line_color = ['blue', 'red', 'black', 'cyan']
    # line_thickness = ['1px', '2px', '3px']

    data = {}
    data['tables'] = []

    for j, combo in enumerate(itertools.product(line_thickness, line_type, line_color)):
        css = "table, th, td, tr {\n" \
              "\tborder: "
        params = []
        for i, x in enumerate(combo):
            params.append(x)
            css += x
            css += ' '

        css += '; \n' \
               '}'

        print(j, css)
        filename = str(j) + ".css"

        fw = open("./CSS/"+str(filename), "w+")
        fw.write(str(css))
        fw.close()

        lth = params[0]
        lty = params[1]
        lclr = params[2]
        table_str_json = build_json(j, lth, lty, lclr)
        with open('./JSON/' + str(j) + '_str.json', 'w') as outfile:
            json.dump(table_str_json, outfile)


#main()
load_params()
print(parameters)