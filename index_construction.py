"""
Main program for index construction in the information retrieval system.
Supports BSBI and SPIMI indexing methods, as well as various compression algorithms.
"""

import argparse
import os
import sys

from bsbi_index import BSBIIndex
from spimi_index import SPIMIIndex
from lsi_index import LSIIndex
from compression import StandardPostings, VBEPostings, OptPForDeltaPostings

def main():
    """ 
    Main function to handle command line arguments and start the indexing process.
    Supports building standard indices as well as LSI (Latent Semantic Indexing) indices.
    """
    parser = argparse.ArgumentParser(description='Index Construction for Information Retrieval System')
    
    parser.add_argument('--method', type=str, choices=['bsbi', 'spimi'], default='spimi',
                        help='Indexing method: bsbi or spimi (default: spimi)')
    
    parser.add_argument('--compression', type=str, choices=['standard', 'vbe', 'optpfordelta'], default='vbe',
                        help='Compression algorithm: standard, vbe, or optpfordelta (default: vbe)')
    
    parser.add_argument('--lsi', action='store_true',
                        help='Build LSI index after main indexing is finished')
    parser.add_argument('--lsi_k', type=int, default=100,
                        help='Number of dimensions for LSI (default: 100)')
    
    parser.add_argument('--data_dir', type=str, default='collection',
                        help='Directory containing document collection (default: collection)')
    parser.add_argument('--output_dir', type=str, default='index',
                        help='Directory for storing index results (default: index)')
    
    parser.add_argument('--memory_threshold', type=int, default=10,
                        help='Memory threshold in MB for SPIMI (default: 10)')

    args = parser.parse_args()

    compression_map = {
        'standard': StandardPostings,
        'vbe': VBEPostings,
        'optpfordelta': OptPForDeltaPostings
    }
    postings_encoding = compression_map[args.compression]

    if args.method == 'bsbi':
        print(f"Building index using BSBI with {args.compression} compression...")
        indexer = BSBIIndex(data_dir=args.data_dir, 
                           postings_encoding=postings_encoding, 
                           output_dir=args.output_dir)
    else:
        print(f"Building index using SPIMI with {args.compression} compression...")
        indexer = SPIMIIndex(data_dir=args.data_dir, 
                             postings_encoding=postings_encoding, 
                             output_dir=args.output_dir,
                             memory_threshold_mb=args.memory_threshold)

    indexer.index()
    print("Main indexing finished.")

    if args.lsi:
        print(f"Building LSI index with k={args.lsi_k}...")
        lsi_indexer = LSIIndex(data_dir=args.data_dir, 
                               output_dir=args.output_dir, 
                               postings_encoding=postings_encoding)
        lsi_indexer.build_lsi(k=args.lsi_k)
        lsi_indexer.save_lsi()
        print("LSI indexing finished.")

if __name__ == "__main__":
    main()
