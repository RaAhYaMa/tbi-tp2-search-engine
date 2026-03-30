import argparse
import os
import sys

from bsbi_index import BSBIIndex
from spimi_index import SPIMIIndex
from lsi_index import LSIIndex
from compression import StandardPostings, VBEPostings, OptPForDeltaPostings

def main():
    parser = argparse.ArgumentParser(description='Index Construction for Information Retrieval System')
    
    # Method selection
    parser.add_argument('--method', type=str, choices=['bsbi', 'spimi'], default='spimi',
                        help='Indexing method: bsbi or spimi (default: spimi)')
    
    # Compression selection
    parser.add_argument('--compression', type=str, choices=['standard', 'vbe', 'optpfordelta'], default='vbe',
                        help='Compression algorithm: standard, vbe, or optpfordelta (default: vbe)')
    
    # LSI option
    parser.add_argument('--lsi', action='store_true',
                        help='Build LSI index after main indexing')
    parser.add_argument('--lsi_k', type=int, default=100,
                        help='Number of dimensions for LSI (default: 100)')
    
    # Directories
    parser.add_argument('--data_dir', type=str, default='collection',
                        help='Directory containing the collection (default: collection)')
    parser.add_argument('--output_dir', type=str, default='index',
                        help='Directory to store the index (default: index)')
    
    # SPIMI specific
    parser.add_argument('--memory_threshold', type=int, default=10,
                        help='Memory threshold in MB for SPIMI (default: 10)')

    args = parser.parse_args()

    # Map compression string to class
    compression_map = {
        'standard': StandardPostings,
        'vbe': VBEPostings,
        'optpfordelta': OptPForDeltaPostings
    }
    postings_encoding = compression_map[args.compression]

    # Map method to indexer
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

    # Perform indexing
    indexer.index()
    print("Main indexing completed.")

    # Build LSI index if requested
    if args.lsi:
        print(f"Building LSI index with k={args.lsi_k}...")
        lsi_indexer = LSIIndex(data_dir=args.data_dir, 
                               output_dir=args.output_dir, 
                               postings_encoding=postings_encoding)
        lsi_indexer.build_lsi(k=args.lsi_k)
        lsi_indexer.save_lsi()
        print("LSI indexing completed.")

if __name__ == "__main__":
    main()
