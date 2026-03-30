"""
Contoh skrip sederhana untuk mendemonstrasikan fungsionalitas pencarian menggunakan TF-IDF.
Skrip ini mengasumsikan indeks sudah dibangun sebelumnya.
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
    print("Hasil  :")
    for (score, doc) in index_instance.retrieve_tfidf(query, k = 10):
        print(f"  {doc:30} {score:>.3f}")
    print()
