RST_FILES = deRham_diagrams.rst exteriorDerivatives.rst projections.rst hodge.rst tests.rst code_units.rst particleMesh.rst TimeLoops.rst deRhamComplex.rst GoogleTest.rst CodeStyle.rst InputFiles.rst

%.rst: %.tex
	pandoc $< --mathjax --columns=120 --wrap=preserve --bibliography spectral_gempic.bib --citeproc -o $@
	python3 convertEq.py $@

all: $(RST_FILES)
	
clean:
	rm -f *.rst *.aux *.bbl *.fls *.out *.toc *.log *.gz *.fdb_latexmk *.blg
