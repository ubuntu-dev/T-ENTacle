from KE_class import KnowledgeExtractor
import itertools
from collections import Counter
import copy
from nltk import ngrams
from knowledge_facilitator import cli

# KE = KnowledgeExtractor()
#
# line = ["ATP : 23 psi", "Avg press: 60 psi",
#         "ATP: 80 psi", "MTP: 70 psi"]
#
# # print(KE.break_natural_boundaries("ATP: 25.6 psi"))
# KE.create_patterns_per_doc(line, "garbage")


# def remove_subpatterns(counted_patterns):
#     """
#
#     :param counted_patterns: a collections.Counter object
#     :return:
#     """
#     counted_copy = copy.deepcopy(counted_patterns)
#     for pattern in counted_patterns.keys():
#         #create all n-gram subpatterns
#         subpatterns = [list(ngrams(pattern, i)) for i in range(1, len(pattern))]
#         print(subpatterns)
#         for subpat in subpatterns:
#             print(subpat)
#             if subpat in counted_copy.keys() and counted_copy[subpat] == counted_copy[pattern]:
#                 counted_copy.pop(subpat)
#
#     print("final patterns ", counted_copy)
#
# pat_counts = Counter(line)
# print(pat_counts)
# remove_subpatterns(pat_counts)
#python knowledge_facilitator.py --i --f --e entity_name.txt /Users/akdidier/Documents/Drilling-Grant/pdfs/Anadarko/AVALANCHE_29-40_UNIT_1H_Completion_Reports.pdf

cli("/Users/akdidier/Documents/Drilling-Grant/pdfs/Anadarko/AVALANCHE_29-40_UNIT_1H_Completion_Reports.pdf", True, False, True, "entity_name.txt")