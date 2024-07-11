#!/bin/bash

cd doxygen
doxygen Doxyfile.in
cd ../sphinx
make html
xdg-open _build/html/index.html
