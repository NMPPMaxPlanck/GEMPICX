RST_FILES = Maxwell_with_discrete_forms.rst multiDdeRhamComplex.rst tests.rst code_units.rst forms_and_operators.rst particleMesh.rst TimeLoops.rst GoogleTest.rst CodeStyle.rst InputFiles.rst ContributingToDoc.rst

%.rst: %.tex
	pandoc $< --mathjax --columns=120 --wrap=preserve --bibliography spectral_gempic.bib --citeproc -o $@
	python3 convertEq.py $@

all: $(RST_FILES)
	
clean:
	rm -f *.rst *.aux *.bbl *.fls *.out *.toc *.log *.gz *.fdb_latexmk *.blg
