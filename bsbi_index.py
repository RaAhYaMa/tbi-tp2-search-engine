import os
import contextlib
from tqdm import tqdm

from base_index import BaseIndex
from index import InvertedIndexReader, InvertedIndexWriter
from util import preprocess

class BSBIIndex(BaseIndex):
    """
    Implementasi algoritma Blocked Sort-Based Indexing (BSBI).
    Membagi koleksi dokumen ke dalam blok-blok berdasarkan sub-direktori untuk 
    memudahkan proses indexing pada memori yang terbatas.
    """
    def parse_block(self, block_dir_relative):
        """
        Membaca blok (sub-direktori) dan mengekstrak pasangan (term_id, doc_id).

        Args:
            block_dir_relative (str): Path relatif sub-direktori blok di dalam data_dir.

        Returns:
            list: List of tuple (term_id, doc_id) dari semua dokumen dalam blok tersebut.
        """
        dir_path = os.path.join(self.data_dir, block_dir_relative)
        td_pairs = []
        for filename in next(os.walk(dir_path))[2]:
            docname = os.path.join(dir_path, filename)
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                for token in preprocess(f.read()):
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))
        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversi pada pasangan term-document dan menuliskannya ke indeks.

        Args:
            td_pairs (list): List of tuple (term_id, doc_id).
            index (InvertedIndexWriter): Objek writer untuk menulis hasil data yang dibalik.
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            term_tf[term_id][doc_id] = term_tf[term_id].get(doc_id, 0) + 1
        
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def index(self):
        """
        Menjalankan seluruh proses pembuatan indeks BSBI.
        Proses ini mencakup pembacaan blok, pembuatan indeks antara, penyimpanan Kamus (Dictionary),
        dan penggabungan (merging) indeks antara menjadi indeks akhir.
        """
        self.intermediate_indices = []
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_' + block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index_writer:
                self.invert_write(td_pairs, index_writer)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)

if __name__ == "__main__":
    from compression import VBEPostings
    BSBI_instance = BSBIIndex(data_dir='collection', 
                              postings_encoding=VBEPostings, 
                              output_dir='index')
    BSBI_instance.index()
