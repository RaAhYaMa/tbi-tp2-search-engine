"""
Program utama untuk membangun indeks (index construction) pada sistem temu balik informasi.
Mendukung metode indexing BSBI dan SPIMI, serta berbagai algoritma kompresi.
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
    Fungsi utama untuk menangani argumen baris perintah dan memulai proses indexing.
    Mendukung pembangunan indeks standar serta indeks LSI (Latent Semantic Indexing).
    """
    parser = argparse.ArgumentParser(description='Pembangunan Indeks untuk Sistem Temu Balik Informasi')
    
    parser.add_argument('--method', type=str, choices=['bsbi', 'spimi'], default='spimi',
                        help='Metode indexing: bsbi atau spimi (default: spimi)')
    
    parser.add_argument('--compression', type=str, choices=['standard', 'vbe', 'optpfordelta'], default='vbe',
                        help='Algoritma kompresi: standard, vbe, atau optpfordelta (default: vbe)')
    
    parser.add_argument('--lsi', action='store_true',
                        help='Bangun indeks LSI setelah indexing utama selesai')
    parser.add_argument('--lsi_k', type=int, default=100,
                        help='Jumlah dimensi untuk LSI (default: 100)')
    
    parser.add_argument('--data_dir', type=str, default='collection',
                        help='Direktori yang berisi koleksi dokumen (default: collection)')
    parser.add_argument('--output_dir', type=str, default='index',
                        help='Direktori untuk menyimpan hasil indeks (default: index)')
    
    parser.add_argument('--memory_threshold', type=int, default=10,
                        help='Ambang batas memori dalam MB untuk SPIMI (default: 10)')

    args = parser.parse_args()

    compression_map = {
        'standard': StandardPostings,
        'vbe': VBEPostings,
        'optpfordelta': OptPForDeltaPostings
    }
    postings_encoding = compression_map[args.compression]

    if args.method == 'bsbi':
        print(f"Membangun indeks menggunakan BSBI dengan kompresi {args.compression}...")
        indexer = BSBIIndex(data_dir=args.data_dir, 
                           postings_encoding=postings_encoding, 
                           output_dir=args.output_dir)
    else:
        print(f"Membangun indeks menggunakan SPIMI dengan kompresi {args.compression}...")
        indexer = SPIMIIndex(data_dir=args.data_dir, 
                             postings_encoding=postings_encoding, 
                             output_dir=args.output_dir,
                             memory_threshold_mb=args.memory_threshold)

    indexer.index()
    print("Indexing utama selesai.")

    if args.lsi:
        print(f"Membangun indeks LSI dengan k={args.lsi_k}...")
        lsi_indexer = LSIIndex(data_dir=args.data_dir, 
                               output_dir=args.output_dir, 
                               postings_encoding=postings_encoding)
        lsi_indexer.build_lsi(k=args.lsi_k)
        lsi_indexer.save_lsi()
        print("Indexing LSI selesai.")

if __name__ == "__main__":
    main()
