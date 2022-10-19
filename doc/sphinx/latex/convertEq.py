import re
import sys
"""_summary_
 Completes conversion of .tex file into .rst file for proper equation numbering and referencing.
 - \label is replaced by :label: (keeping actual label)
 - eqref is replaced by :eq:
    
    Reminder on regexp class:
    information about the match object are availble in the methods
    .span() returns a tuple containing the start-, and end positions of the match.
    .string returns the string passed into the function
    .group() returns the part of the string where there was a match
"""

def convert(file):
    f = open(file,'r')
    flines = f.readlines()

    # replace \label by :label: and switch empty line from above to below
    for i in range(flines.__len__()):
        line = flines[i]
        hasLabel = re.search(r'\\label',line)
        if hasLabel:
            line = re.sub(r"\\label*{",":label: ", line)
            line = re.sub("\\}","", line)
            flines[i-1] = line
            flines[i] = "\n"
    
    # replace \eq by :eq:            
    for i in range(flines.__len__()):
        line = flines[i]       
        hasEqref = re.search("`\[.*\]",line)
        if hasEqref:
            u = re.split("__", line)
            lineout = ""
            for s in u:
                x = re.search("`.*`", s)
                if x:
                    xx = re.search("<.*>", s)
                    y = s.replace(x.group(),r":eq:`"+xx.group()[2:-1]+r"` ")
                    lineout = lineout + y 
                else:
                    lineout = lineout + s
            flines[i] = lineout

    f.close()
    fout = open(file,'w')
    fout.writelines(flines)

if __name__ == '__main__':
    file = sys.argv[1]
    # Check that it is a .rst file
    if (file[-4:] != '.rst'):
        print('wrong file format. convertEq should be applied to a .rst file')
        print('input file name is ' + file) 
    convert(file)
