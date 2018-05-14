import knowledge_extractor
from stats import run_comparison

knowledge_extractor.create_csv("ARABIAN 30-19 FED COM 3H Well History.pdf")
run_comparison("report.csv")