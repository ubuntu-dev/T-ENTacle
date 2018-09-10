
import random
import os
from shutil import copy, rmtree
import subprocess
import time
import json

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

# python knowledge_facilitator.py --i --f --e entity_name_test.txt /Users/akdidier/Documents/T-ENTacle/test_docs/
# cli("/Users/akdidier/Documents/T-ENTacle/test_docs/", True, False, True, "entity_name_test.txt")

base_dir = "/Users/akdidier/Documents/Drilling-Grant/pdfs"
pdf_dirs = ["Anadarko", "Cimarex", "EOG"]

random.seed(12345)

sample_sizes = range(1,7)
times = []
for size in sample_sizes:
    files = []
    for dir in pdf_dirs:
        path = os.path.join(base_dir, dir)
        selection = random.sample(os.listdir(path), size)
        selection = [os.path.join(path, f) for f in selection]
        files.extend(selection)

    test_dir = os.path.join(os.getcwd(), "training_files")
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    for f in files:
        copy(f, test_dir)
    print("Training the model on " + str(size) + " files from each vendor")
    start = time.time()
    subprocess.run(["python", "knowledge_facilitator.py", "--i", "--f", "--e", "entity_name.txt", test_dir])
    end = time.time()
    runtime = end - start
    times.append(runtime)
    print("Finished training model for this iteration. Run time: ", runtime)
    #rename the files
    learned_patterns_old = os.path.join(os.getcwd(), "model/learned_patterns.pkl")
    learned_patterns_new = os.path.join(os.getcwd(), "model/learned_patterns_allvendors_"+str(size)+".pkl")
    os.rename(learned_patterns_old, learned_patterns_new)
    all_patterns_old = os.path.join(os.getcwd(), "model/all_patterns.pkl")
    all_patterns_new = os.path.join(os.getcwd(), "model/all_patterns_allvendors_" + str(size) + ".pkl")
    os.rename(all_patterns_old, all_patterns_new)
    rmtree(test_dir)

time_info = {"size": sample_sizes, "times": times}
with open("time_info.json", "w") as f:
    json.dump(time_info, f)