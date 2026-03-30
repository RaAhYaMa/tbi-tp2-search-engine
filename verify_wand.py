"""
Skrip untuk memverifikasi kebenaran algoritma WAND (Weak AND) dibandingkan dengan 
pencarian BM25 standar (exhaustive search). Membandingkan hasil dokumen, skor, dan waktu eksekusi.
"""

import sys
import time
from base_index import BaseIndex
from compression import VBEPostings

def verify():
    """ 
    Melakukan pengujian komparatif antara retrieve_bm25 dan retrieve_bm25_wand.
    Memastikan bahwa optimasi WAND memberikan hasil yang sama dengan pencarian penuh 
    namun dengan performa yang lebih cepat.
    """
    bsbi = BaseIndex(data_dir='collection', 
                     postings_encoding=VBEPostings, 
                     output_dir='index')
    
    test_queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy"
    ]

    print(f"{'Query':<40} | {'Status':<10} | {'Waktu Asli':<11} | {'Waktu WAND':<11} | {'Speedup':<9} | {'Keterangan'}")
    print("-" * 115)

    for query in test_queries:
        start_orig = time.perf_counter()
        res_original = bsbi.retrieve_bm25(query, k=10)
        end_orig = time.perf_counter()
        time_orig = end_orig - start_orig

        start_wand = time.perf_counter()
        res_wand = bsbi.retrieve_bm25_wand(query, k=10)
        end_wand = time.perf_counter()
        time_wand = end_wand - start_wand

        speedup = time_orig / time_wand if time_wand > 0 else 0

        if len(res_original) != len(res_wand):
            print(f"{query[:40]:<40} | GAGAL      | {time_orig:10.4f}s | {time_wand:10.4f}s | {speedup:8.2f}x | Beda jumlah: {len(res_original)} vs {len(res_wand)}")
            continue
        match = True
        max_diff = 0
        for i in range(len(res_original)):
            score_orig, doc_orig = res_original[i]
            score_wand, doc_wand = res_wand[i]

            if doc_orig != doc_wand:
                match = False
                break
            
            diff = abs(score_orig - score_wand)
            if diff > max_diff:
                max_diff = diff

        if match and max_diff < 1e-4: 
            print(f"{query[:40]:<40} | BERHASIL   | {time_orig:10.4f}s | {time_wand:10.4f}s | {speedup:8.2f}x | Selisih skor: {max_diff:.6f}")
        else:
            print(f"{query[:40]:<40} | GAGAL      | {time_orig:10.4f}s | {time_wand:10.4f}s | {speedup:8.2f}x | Dokumen beda atau selisih besar")

if __name__ == "__main__":
    verify()
