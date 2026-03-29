import sys
import time
from base_index import BaseIndex
from compression import VBEPostings

def verify():
    # Inisialisasi index
    bsbi = BaseIndex(data_dir='collection', 
                     postings_encoding=VBEPostings, 
                     output_dir='index')
    
    # Kumpulan query uji
    test_queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy"
    ]

    print(f"{'Query':<40} | {'Status':<10} | {'Org Time':<11} | {'WAND Time':<11} | {'Speedup':<9} | {'Notes'}")
    print("-" * 115)

    for query in test_queries:
        # Panggil kedua metode
        start_orig = time.perf_counter()
        res_original = bsbi.retrieve_bm25(query, k=10)
        end_orig = time.perf_counter()
        time_orig = end_orig - start_orig

        start_wand = time.perf_counter()
        res_wand = bsbi.retrieve_bm25_wand(query, k=10)
        end_wand = time.perf_counter()
        time_wand = end_wand - start_wand

        speedup = time_orig / time_wand if time_wand > 0 else 0

        # 1. Cek jumlah hasil
        if len(res_original) != len(res_wand):
            print(f"{query[:40]:<40} | FAILED     | {time_orig:10.4f}s | {time_wand:10.4f}s | {speedup:8.2f}x | Diff length: {len(res_original)} vs {len(res_wand)}")
            continue

        # 2. Bandingkan isi (doc_id dan score)
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

        if match and max_diff < 1e-4: # Toleransi floating point
            print(f"{query[:40]:<40} | SUCCESS    | {time_orig:10.4f}s | {time_wand:10.4f}s | {speedup:8.2f}x | Max diff: {max_diff:.6f}")
        else:
            print(f"{query[:40]:<40} | FAILED     | {time_orig:10.4f}s | {time_wand:10.4f}s | {speedup:8.2f}x | Doc mismatch or large diff")

if __name__ == "__main__":
    verify()
