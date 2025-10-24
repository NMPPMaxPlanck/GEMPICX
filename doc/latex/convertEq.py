import re
import sys
r"""_summary_
 Completes conversion of .tex file into .rst file for proper equation numbering and referencing, so that
 - a \label{*} in a math environment is replaced by :label: (keeping the actual label *)
 - a reference to {sec:*} is replaced by :ref: (keeping the actual reference *)
 - a reference to {eq:*} is replaced by :eq: (also keeping the actual reference *)
    
    Reminder on regexp class:
    information about the match object are available in the methods
    .span() returns a tuple containing the start-, and end positions of the match.
    .string returns the string passed into the function
    .group() returns the part of the string where there was a match
"""

def convert(file):

    with open(file,'r', encoding="utf-8") as f:
        flines = f.readlines()

    # replace \label{*} by :label:* and switch empty line from above to below
    for i in range(flines.__len__()):
        line = flines[i]
        hasLabel = re.search(r'\\label',line)
        if hasLabel:
            line = re.sub(r"\\label*{",":label: ", line)
            line = re.sub("\\}","", line)
            flines[i-1] = line
            flines[i] = "\n"

    # LaTeX and corresponding Sphinx tags
    labelTags = {'eq': ':eq:', 'sec': ':ref:'}
    # Sphinx tag for everything else, e.g. fig:, tab:, no LaTeX tag ...
    defaultTag = ':numref:'
    # captures any rst-references of the standard pandoc form:
    # example text with a reference: `[labeltxt] <#labeltxt>`__
    stdRef=r"(`\[(.*?)\] <#\2>`__)"
    # captures any commented rst-references of the form:
    # example text with a reference: `shown reference text <#labeltxt>`__
    commentRef=r"(`([^`\[]*) <#(.*?)>`__)"

    # Replace pandoc generated references with Sphinx references so links and numbering works
    for i in range(flines.__len__()):
        line = flines[i]               

        for pattern, label in re.findall(stdRef, line):
            tag = label.split(':')[0]
            sphinxTag = labelTags.get(tag, defaultTag)
            replacement = rf"{sphinxTag}`{label}`"
            line = re.sub(re.escape(pattern), replacement, line)
        for pattern, referenceText, label in re.findall(commentRef, line):
            tag = label.split(':')[0]
            sphinxTag = labelTags.get(tag, defaultTag)
            if re.fullmatch(r"[\d.]*", referenceText):
                # Reference text is pandoc generated numbering, which we want Sphinx to do
                replacement = rf"{sphinxTag}`{label}`"
            else:
                # Reference text could be relevant, so we keep it
                replacement = rf"{sphinxTag}`{referenceText} <{label}>`"
            line = re.sub(re.escape(pattern), replacement, line)
        
        # pandoc 3 also doesn't handle figures properly, so we have to fix the tag line
        # `name: fig:...` --> `:name: fig:...`
        line = re.sub(r"([^:])(name: fig:)", r"\1:\2", line)
            
        flines[i] = line

    # pandoc also doesn't handle tables properly, so we have to shift the tag line into the table
    flines = ''.join(flines)
    flines = re.sub(r"(.. container::\n)( *:name: tab:.*\n)(\n *.. table.*\n)", r"\1\3   \2", flines)

    with open(file,'w', encoding='utf-8', newline='\n') as fout:
        fout.writelines(flines)

if __name__ == '__main__':
    file = sys.argv[1]
    # Check that it is a .rst file
    if (file[-4:] != '.rst'):
        print('wrong file format. convertEq should be applied to a .rst file')
        print('input file name is ' + file) 
    convert(file)
