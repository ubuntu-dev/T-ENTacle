# T-ENTacle
Extracting structured information from PDFs - and later, diagrams. 
Here, we create entity relationships using both text analysis and stucture analysis. 
These two processes are currently seperate but will need to be integrated in the future to provide accurate entity relationships

## Getting Started

In order to get the code running, you will need to have python3 installed. You will then need to download the dependencies. Clone this repo and then run the following inside of the top level directory:
```
pip install -r requirements.txt
```

### Files

structure-analysis/
- Folder contains modications to pdftabextract found here:
https://github.com/WZBSocialScienceCenter/pdftabextract

text-analysis/
- Folder contains much of the text-analysis using various NLP methods
- knowledge-extractor.py identitfies statistically significant n-grams that are then filtered

### Running Code
In order to run the knowledge extractor, you need to change the second to last line of knowledge-extractor.py that is commented out:

```
pdf_buffer=create_csv('/PATH/YOUR_FILE.pdf')
```

Run using this:
```
python knowledge_extractor.py
```
This will generate a csv file that contains the extracted entities. 
