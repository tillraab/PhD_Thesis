TEXBASE=discussion

TEXFILE=$(TEXBASE).tex
PDFFILE=$(TEXBASE).pdf
# REVISION=b7b3d179633664316f7e60df9629b90c14b653e0
REVISION=74feb0945cacecaeec88387c42a20948f299e3ba

all: bib

bib: $(TEXBASE).bbl
$(TEXBASE).bbl: $(TEXFILE) references.bib
	pdflatex $(TEXFILE)
	bibtex $(TEXBASE)
	pdflatex $(TEXFILE)
	pdflatex $(TEXFILE)
	pdflatex $(TEXFILE)
	@echo
	@echo "BibTeX log:"
	@sed -n -e '1,/You.ve used/p' $(TEXBASE).blg

pdf: $(PDFFILE)
$(PDFFILE) : $(TEXFILE)
	pdflatex -interaction=scrollmode $< | tee /dev/stderr | fgrep -q "Rerun to get cross-references right" && pdflatex -interaction=scrollmode $< || true

again :
	pdflatex $(TEXFILE)


diff :
	#latexdiff-git -r $(REVISION) --pdf $(TEXFILE)
	latexdiff-git -r $(REVISION) $(TEXFILE)
	-pdflatex $(TEXBASE)-diff$(REVISION)
	-bibtex $(TEXBASE)-diff$(REVISION)
	-pdflatex $(TEXBASE)-diff$(REVISION)
	-pdflatex $(TEXBASE)-diff$(REVISION)
	-pdflatex $(TEXBASE)-diff$(REVISION)
	mv $(TEXBASE)-diff$(REVISION).pdf $(TEXBASE)-diff.pdf
	rm $(TEXBASE)-diff$(REVISION).*

stats: $(PDFFILE)
# use \pagestyle{empty} and don't include any pictures!
	pdftotext -nopgbrk $(PDFFILE) - | fold -s > tmp.txt
	@echo
	@echo "     words: " `wc -w tmp.txt 2> /dev/null | cut -d ' ' -f 1` 
	@echo "characters: " `wc -c tmp.txt 2> /dev/null | cut -d ' '  -f 1`
	rm tmp.txt

figures:
	./latexfigures2png $(TEXFILE)

watchpdf :
	while true; do ! make -q pdf && make pdf; sleep 0.5; done

rtf :
	latex2rtf $(TEXFILE)

# remove all fancy commands from the tex file:
simplify :
	sed -e '/overall style/,/page style/d; /setdoublespacing/,+1d; /usepackage.*caption/s/{\(.*\)}/\1/; /figure placement/,/^%/d; /ifthenelse.*nofigs/,/#1/d; /begin{multicols}/d; /end{multicols}/d; /begin{keywords}/,/end{keywords}/d; /begin{contributions}/,/end{contributions}/d; /figurecaptions/d; /linenomath/d; s/captionc/caption/' $(TEXFILE) | perl -00 -lpe 's/\\showfigure\{((\s|.)*?)\}/$$1/' > $(TEXBASE)-simplified.tex

clean:
	rm -rf auto *~ *.aux *.blg *.bbl *.dvi *.log *.out
	rm -f $(TEXBASE).rtf $(TEXBASE)-simplified.*
	rm -f figure-??.png figure-??.jpg

cleanall: clean
	rm -f $(PDFFILE)
