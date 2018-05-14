Extraction of well-defined (well marked columns and rows) tables that span most of the page. We deal with this in the following way:  
a.       Do OCR (using Poppler software called Pdftohtml https://en.wikipedia.org/wiki/Poppler_(software) )  
b.       Using the Open CV library (in python) to detect well defined columns and rows and hence creating grids.  
c.       Mapping the OCR text to the relevant grids and hence producing a csv.  
d.       The above attributes all the text from a pdf page to some grid regardless of the fact that it is a table or boilerplate around the table.  
e.       The task remains to actually then extract the table from the rest of the boilerplate on the page/csv (we are working on it)  

Extraction of non-marked tables (with no explicit lines for columns and rows) that span most of the page:  
a.       Do OCR.  
b.       Do Grid (Open CV, python) recognition using “white spaces” as clues.  
c.       Map Text to grid.  
d.       Again, this fits the whole page with grids as before.  

Extraction of tables or tabular looking structures (marked or unmarked) that are in a small part of the page or multiple small tabular structures of different kinds in the same page. This is a much harder problem, but we are trying to solve part of this using a “Divide and Conquer” strategy.  
a.       Do OCR.  
b.       Detect the largest explicit structures/lines and segment the whole page according to them.  
c.       Apply the above techniques (in # 1 and # 2) to detect tabular structures in each of the segments.  
