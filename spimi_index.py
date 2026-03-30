import os
import sys
import contextlib
from tqdm import tqdm

from base_index import BaseIndex
from index import InvertedIndexReader, InvertedIndexWriter
from util import preprocess

class SPIMIIndex(BaseIndex):
    """
    Implementasi Single Pass In-Memory Indexing.
    Membagi blok berdasarkan penggunaan memori (threshold).
    """
    def __init__(self, data_dir, output_dir, postings_encoding, 
                 memory_threshold_mb=10, index_name="main_index"):
        super().__init__(data_dir, output_dir, postings_encoding, index_name)
        # Ambang batas dalam bytes
        self.memory_threshold = memory_threshold_mb * 1024 * 1024

    def index(self):
        self.intermediate_indices = []
        block_id = 0
        term_dict = {} # term_id -> set of doc_ids
        term_tf = {}   # term_id -> {doc_id -> freq}
        
        # Iterasi global semua file di dalam folder data_dir
        all_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                all_files.append(os.path.join(root, file))

        for doc_path in tqdm(sorted(all_files)):
            doc_id = self.doc_id_map[doc_path]
            
            with open(doc_path, "r", encoding="utf8", errors="surrogateescape") as f:
                # Preprocessing lengkap (case folding, tokenization, stopword removal, stemming)
                tokens = preprocess(f.read())
                
            for token in tokens:
                term_id = self.term_id_map[token]
                
                if term_id not in term_dict:
                    term_dict[term_id] = set()
                    term_tf[term_id] = {}
                
                term_dict[term_id].add(doc_id)
                term_tf[term_id][doc_id] = term_tf[term_id].get(doc_id, 0) + 1

            # Estimasi kasarnya penggunaan memori (dictionary + set + nested dict)
            # Karena sys.getsizeof tidak rekursif, kita estimasi lewat jumlah token/entry jika ingin lebih akurat
            # Namun untuk kemudahan, kita coba limit berbasis byte dari dictionary utama
            if (sys.getsizeof(term_dict) + sys.getsizeof(term_tf)) > self.memory_threshold:
                self.flush_block(term_dict, term_tf, block_id)
                term_dict.clear()
                term_tf.clear()
                block_id += 1

        if term_dict:
            self.flush_block(term_dict, term_tf, block_id)

        self.save()

        # Merging semua hasil intermediate
        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)

    def flush_block(self, term_dict, term_tf, block_id):
        """Menulis isi memori ke file index intermediate"""
        index_id = f'spimi_block_{block_id}'
        self.intermediate_indices.append(index_id)
        with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as writer:
            for term_id in sorted(term_dict.keys()):
                sorted_docs = sorted(list(term_dict[term_id]))
                assoc_tf = [term_tf[term_id][d] for d in sorted_docs]
                writer.append(term_id, sorted_docs, assoc_tf)

if __name__ == "__main__":
    from compression import VBEPostings
    SPIMI_instance = SPIMIIndex(data_dir='collection', 
                                postings_encoding=VBEPostings, 
                                memory_threshold_mb=10, 
                                output_dir='index')
    SPIMI_instance.index()
