"""
Command line interface (CLI) for performing searches in the information retrieval system.
Supports various scoring methods such as TF-IDF, BM25, WAND, and LSI.
"""

import argparse
import sys
from base_index import BaseIndex
from compression import VBEPostings
from lsi_index import LSIIndex

def main():
    """ 
    Main function to handle command line arguments and display search results.
    Can accept a single query or a list of queries from a file.
    """
    parser = argparse.ArgumentParser(description="Search CLI for Information Retrieval System.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("query", nargs="?", help="Search query text.")
    group.add_argument("--file", "-f", help="Path to the file containing the list of queries (one per line).")
    
    parser.add_argument("-k", type=int, default=10, help="Number of results to return (default: 10).")
    parser.add_argument("--method", "-m", choices=["tfidf", "bm25", "bm25_wand", "lsi"], default="tfidf",
                        help="Scoring method used (default: tfidf).")
    parser.add_argument("--data_dir", default="collection", help="Document collection directory (default: 'collection').")
    parser.add_argument("--output_dir", default="index", help="Index storage directory (default: 'index').")

    args = parser.parse_args()

    if args.method == "lsi":
        index_instance = LSIIndex(data_dir=args.data_dir,
                                  postings_encoding=VBEPostings,
                                  output_dir=args.output_dir)
    else:
        index_instance = BaseIndex(data_dir=args.data_dir,
                                   postings_encoding=VBEPostings,
                                   output_dir=args.output_dir)
    
    if args.file:
        try:
            with open(args.file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
    else:
        queries = [args.query]
 
    for query in queries:
        print(f"Query  : {query}")
        print(f"Method : {args.method}")
        print("Results :")
        
        if args.method == "tfidf":
            results = index_instance.retrieve_tfidf(query, k=args.k)
        elif args.method == "bm25":
            results = index_instance.retrieve_bm25(query, k=args.k)
        elif args.method == "bm25_wand":
            results = index_instance.retrieve_bm25_wand(query, k=args.k)
        elif args.method == "lsi":
            results = index_instance.retrieve_lsi(query, k=args.k)
        else:
            results = []
 
        if not results:
            print("  No results found.")
        else:
            for score, doc in results:
                print(f"  {doc:30} {score:>.3f}")
        print("-" * 40)

if __name__ == "__main__":
    main()
