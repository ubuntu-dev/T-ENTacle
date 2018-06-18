#! /usr/bin/python

import itertools


def main():
    # Could read this from files
    #
    line_type = ['dotted', 'solid']
    line_color = ['blue', 'red', 'black', 'cyan']
    line_thickness = ['1px', '2px', '3px']

    for j, combo in enumerate(itertools.product(line_thickness, line_type, line_color)):
        css = "table, th, td, tr {\n" \
              "\tborder: "
        for i, x in enumerate(combo):
            css += x
            css += ' '

        css += '; \n' \
               '}'

        print(j, css)
        filename = str(j) + ".css"

        fw = open("./CSS/"+str(filename), "w+")
        fw.write(str(css))
        fw.close()


main()
