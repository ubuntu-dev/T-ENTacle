Extraction of entities and their values. 

Using Grobid-Quantities, we detect numbers and their associated units. The units are used to determine a set of candidate entities to which the values could belong. For instance, a unit of "psi" would narrow down the possible entities to those dealing with pressure. The appropriate entity then is determined by calculating the Levenshtein distance between the tokens preceding the value and the candidate entity. 

Currently, Grobid-Quantities works imperfectly, so an additional component with an interactive feedback loop to label missed quantities is in development. The combination of Levenshtein and interactive labeling will generate a new training set to update the Grobid-Quantities model. With this update, the model should be more generalizable and better detect not only quantities, but the entity being quantified. 
