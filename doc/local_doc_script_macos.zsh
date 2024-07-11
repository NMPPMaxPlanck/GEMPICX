#!/bin/zsh

cd doxygen
doxygen Doxyfile.in
cd ../sphinx
make latex-conversion
make html
open _build/html/index.html