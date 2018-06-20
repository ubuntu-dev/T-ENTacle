#! /usr/bin/python

import itertools
import json

def build_json(index, lth, lty, lclr):
    data = {}
    data[str(index)] = []
    data[str(index)].append({
        'line_thickness': str(lth),
        'line_type': str(lty),
        'line_color': str(lclr)
    })
    return data

def main():
    # Could read this from files
    #
    line_type = ['dotted', 'solid']
    line_color = ['blue', 'red', 'black', 'cyan']
    line_thickness = ['1px', '2px', '3px']

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


main()
