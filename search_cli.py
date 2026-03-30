"""
Antarmuka baris perintah (CLI) untuk melakukan pencarian pada sistem temu balik informasi.
Mendukung berbagai metode scoring seperti TF-IDF, BM25, WAND, dan LSI.
"""

import argparse
import sys
from base_index import BaseIndex
from compression import VBEPostings
from lsi_index import LSIIndex

def main():
    """ 
    Fungsi utama untuk menangani argumen baris perintah dan menampilkan hasil pencarian.
    Dapat menerima query tunggal atau daftar query dari sebuah file.
    """
    parser = argparse.ArgumentParser(description="Antarmuka Pencarian (Search CLI) untuk Sistem Temu Balik Informasi.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("query", nargs="?", help="Teks query pencarian.")
    group.add_argument("--file", "-f", help="Path ke file yang berisi daftar query (satu per baris).")
    
    parser.add_argument("-k", type=int, default=10, help="Jumlah hasil yang ingin dikembalikan (default: 10).")
    parser.add_argument("--method", "-m", choices=["tfidf", "bm25", "bm25_wand", "lsi"], default="tfidf",
                        help="Metode skoring yang digunakan (default: tfidf).")
    parser.add_argument("--data_dir", default="collection", help="Direktori koleksi dokumen (default: 'collection').")
    parser.add_argument("--output_dir", default="index", help="Direktori penyimpanan indeks (default: 'index').")

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
            print(f"Error: File '{args.file}' tidak ditemukan.")
            sys.exit(1)
    else:
        queries = [args.query]
 
    for query in queries:
        print(f"Query  : {query}")
        print(f"Metode : {args.method}")
        print("Hasil  :")
        
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
            print("  Tidak ada hasil yang ditemukan.")
        else:
            for score, doc in results:
                print(f"  {doc:30} {score:>.3f}")
        print("-" * 40)

if __name__ == "__main__":
    main()
