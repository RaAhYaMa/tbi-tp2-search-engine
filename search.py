"""
Simple script example to demonstrate search functionality using TF-IDF.
This script assumes the index has already been built.
"""

from base_index import BaseIndex
from compression import VBEPostings

index_instance = BaseIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')
queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]
           
for query in queries:
    print(f"Query  : {query}")
    print("Results :")
    for (score, doc) in index_instance.retrieve_tfidf(query, k = 10):
        print(f"  {doc:30} {score:>.3f}")
    print()
