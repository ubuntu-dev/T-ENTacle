Extraction of entities and their values. 

Using Grobid-Quantities, we detect numbers and their associated units. The units are used to determine a set of candidate entities to which the values could belong. For instance, a unit of "psi" would narrow down the possible entities to those dealing with pressure. The appropriate entity then is determined by calculating the Levenshtein distance between the tokens preceding the value and the candidate entity. 

Currently, Grobid-Quantities works imperfectly, so an additional component with an interactive feedback loop to label missed quantities is in development. The combination of Levenshtein and interactive labeling will generate a new training set to update the Grobid-Quantities model. With this update, the model should be more generalizable and better detect not only quantities, but the entity being quantified. 

# Knowledge Facilitator:

## Follow the instructions to get the CLI running in your local:
Make sure you have python3 installed  
Clone this Git Repo:  
> git clone https://github.com/nasa-jpl/T-ENTacle.git  
  
Install all the dependencies (python libraries):  
> cd T-ENTacle  
> pip3 install -r requirements.txt   
  
## Run the CLI:
> cd text-analysis  
  
For help with the flags and options run:
> python knowledge_facilitator.py --help

Let us run the CLI in a single file mode:  
> python knowledge_facilitator.py --i --f --e entity_name.txt ‘/path/to/pdf/pdf_file'  
  
Some things to note here are:  
* we use  --f to ‘start fresh’ this flag makes sure that old learned/saved files are deleted 
* entity_name.txt is a file name that you can pass. The file contains the name of the entities that you would like to search (one in each line of the file)
* --i makes sure that the user is involved in the process (Necessary in the beginning to create some ground truth)

The program goes through each entity listed in the entity_name.txt file and tries to engage the user in learning the right patterns.
Now we will try to run in batch mode with partial supervision:
In batch mode we give the CLI a directory path as an input instead of a single file path
> python knowledge_facilitator.py --i --a --e entity_name.txt '/path/to/pdf’  
  
Some things to note here are:  
* we did not use the --f flag here, because we want the CLI to access the previous learning. This helps the CLI to suggest better options to the user
* the --a flag makes sure that the CLI does not bother the user for the entities for which the previously learned patterns are an exact match to some pattern in the current pdf (partial supervision)

Let us now run the CLI in fully automatic mode under the assumption that some ’training’ has been provided to the CLI already (by running it in the interactive mode over a few Pdfs):
> python knowledge_facilitator.py --e entity_name.txt '/path/to/pdf’  
  
Some things to note here are:  
* The CLI in this case will not bother the user at all and will identity exact patterns for the entities based only on the previously learned patterns.

## Important Notes:
* After every interaction the CLI creates a report.csv based on all the previous interactions.
* The user can also use the CLI by not giving the entity list in the beginning. In this case the CLI will ask the user to enter the name of the entity they are searching for.
* Any time during the interaction, the user can also search any entity using value. This functionality can be helpful in the following scenarios:
    * The user already has looked at the pdf and has identified on some page, an entity (say Max Pressure) and a value associated with it. Now, using search by value, the user can just input that one value and the CLI will suggest all the patterns that have that value in at least one of their instances in the whole Pdf.
    * This is also a useful feature for detecting entities whose patterns don’t have an entity name or unit associated with them. Or the entity name represented in the pdf is very dissimilar to the name of the actual entity.

## Interesting findings:
* Even for pdf reports that should be in the same format (eg, from the same vendor or the same report format completed by different individuals), the same entity may be present in different ways in different Pdfs:
    * Different name:
        * Max Pressure : 80.6 psi vs Max Psi : 80.6
    * Different pattern:
        * Max Pressure : 80.6 psi vs Max Pressure 80.6
The software is sensitive to these difference and tries not to assume anything at least when it comes to matching the exact patterns. Although, in terms of suggesting the closest patterns, these differences don’t matter much.


## Known shortcomings:
1. For now the software does not take a floating point number as one single unit.
        eg: ’80.6' is seen as ’80' ‘.' ‘6'
        it affects the CLI in two ways:
    * when a pattern is selected by a user, and they get another option to choose the tokens, they have to select each ’80’, ‘.’ and ‘6’ individually
    * in the report, that is generated the number is outputted as : '80 6’
2. When rerunning a file, the software still creates the patterns, instead of importing them from the saved patterns. 

  
## The diagram shows the User interaction, Process and Data Flow for the KE:
![Flow Diagram](https://github.com/nasa-jpl/T-ENTacle/blob/master/text_analysis/flow.jpg)
  
