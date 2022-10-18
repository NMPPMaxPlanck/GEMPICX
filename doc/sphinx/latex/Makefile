RST_FILES = deRham_diagrams.rst

%.rst: %.tex
	pandoc $< --mathjax --wrap=preserve --bibliography spectral_gempic.bib --citeproc -o $@
	python3 convertEq.py $@

all: $(RST_FILES)
	
clean:
	rm -f *.rst *.aux *.bbl *.fls *.out *.toc *.log *.gz *.fdb_latexmk *.blg
