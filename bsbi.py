import os
import pickle
import contextlib
import heapq
import time
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs, preprocess
from compression import StandardPostings, VBEPostings
from tqdm import tqdm

class BSBIIndex:
    """
    Kelas untuk pembuatan indeks menggunakan algoritma Blocked Sort-Based Indexing (BSBI).
    Meliputi proses parsing dokumen, pembuatan indeks antara, penggabungan indeks,
    dan pencarian peringkat (ranked retrieval) menggunakan TF-IDF atau BM25.
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        """
        Inisialisasi objek BSBIIndex.

        Args:
            data_dir (str): Path ke direktori data dokumen.
            output_dir (str): Path ke direktori output untuk menyimpan file indeks.
            postings_encoding: Objek encoding untuk postings list.
            index_name (str): Nama dasar untuk file indeks yang dihasilkan.
        """
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        self.intermediate_indices = []

    def save(self):
        """
        Menyimpan term_id_map dan doc_id_map ke direktori output dalam format pickle.
        """

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """
        Memuat term_id_map dan doc_id_map dari direktori output.
        """

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Membaca blok (sub-direktori) dan mengekstrak pasangan (term_id, doc_id).

        Args:
            block_dir_relative (str): Path relatif sub-direktori blok di dalam data_dir.

        Returns:
            list: List of tuple (term_id, doc_id) dari semua dokumen dalam blok tersebut.
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                for token in preprocess(f.read()):
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversi pada pasangan term-document dan menuliskannya ke indeks.
        Menggunakan strategi mirip SPIMI dengan kamus besar tunggal untuk setiap blok.

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
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Melakukan penggabungan (merging) beberapa indeks antara menjadi satu indeks tunggal.
        Menggunakan algoritma external merge sort.

        Args:
            indices (list): List dari objek InvertedIndexReader yang akan digabung.
            merged_index (InvertedIndexWriter): Objek writer untuk menyimpan hasil penggabungan.
        """
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)
        for t, postings_, tf_list_ in merged_iter:
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan pencarian peringkat menggunakan skema pembobotan TF-IDF.

        Args:
            query (str): Kalimat query pencarian.
            k (int): Jumlah dokumen teratas yang ingin dikembalikan.

        Returns:
            list: List of tuple (score, doc_name) yang telah diurutkan menurun berdasarkan skor.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        query_tokens = preprocess(query)
        terms = [self.term_id_map[token] for token in query_tokens if token in self.term_id_map]

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: (x[0], x[1]), reverse = True)[:k]

    def retrieve_bm25(self, query, k = 10, k1 = 1.6, b = 0.75):
        """
        Melakukan pencarian peringkat menggunakan skema pembobotan BM25.

        Args:
            query (str): Kalimat query pencarian.
            k (int): Jumlah dokumen teratas yang ingin dikembalikan.
            k1 (float): Parameter k1 (default: 1.6).
            b (float): Parameter b (default: 0.75).

        Returns:
            list: List of tuple (score, doc_name) yang telah diurutkan menurun berdasarkan skor.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        query_tokens = preprocess(query)
        terms = [self.term_id_map[token] for token in query_tokens if token in self.term_id_map]
        
        with InvertedIndexReader(self.index_name,
                                self.postings_encoding,
                                directory=self.output_dir) as merged_index:
            scores = {}
            N = len(merged_index.doc_length)
            avdl = merged_index.avg_doc_length

            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    idf = math.log(N / df)

                    postings, tf_list = merged_index.get_postings_list(term)

                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        
                        dl = merged_index.doc_length[doc_id]

                        tf_weight = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * dl / avdl) + tf)
                        
                        scores[doc_id] += idf * tf_weight

            docs = [(score, self.doc_id_map[doc_id])
                    for (doc_id, score) in scores.items()]

            return sorted(docs, key = lambda x: (x[0], x[1]), reverse = True)[:k]
    
    def retrieve_bm25_wand(self, query, k = 10, k1 = 1.6, b = 0.75):
        """
        Melakukan pencarian peringkat menggunakan algoritma WAND (Weak AND) dengan pembobotan BM25.

        Args:
            query (str): Kalimat query pencarian.
            k (int): Jumlah dokumen teratas yang ingin dikembalikan.
            k1 (float): Parameter k1 (default: 1.6).
            b (float): Parameter b (default: 0.75).

        Returns:
            list: List of tuple (score, doc_name) yang telah diurutkan menurun berdasarkan skor.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        query_tokens = preprocess(query)
        terms = [self.term_id_map[token] for token in query_tokens if token in self.term_id_map]

        with InvertedIndexReader(self.index_name,
                                self.postings_encoding,
                                directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            avdl = merged_index.avg_doc_length
            
            term_data = []
            min_dl = min(merged_index.doc_length.values())
            for term in terms:
                if term in merged_index.postings_dict:
                    data = merged_index.postings_dict[term]
                    max_tf = data[4]
                    df = data[1]
                    idf = math.log(N / df)

                    ut = idf * ((k1 + 1) * max_tf) / (k1 * ((1 - b) + b * min_dl / avdl) + max_tf)

                    postings, tf_list = merged_index.get_postings_list(term)
                    p_length = len(postings)
                    postings.append(-1)
                    term_data.append({'p': postings, 'tf': tf_list, 'idx': 0, 'ut': ut, 'idf': idf, 'p_len': p_length})

            if not term_data: return []

            top_k = []
            threshold = 0
            while True:
                term_data.sort(key=lambda x: x['p'][x['idx']] if x['idx'] < x['p_len'] else float('inf'))

                score_bound = 0.0
                pivot_idx = -1
                for i in range(len(term_data)):
                    score_bound += term_data[i]['ut']
                    if score_bound > threshold:
                        pivot_idx = i
                        break

                if pivot_idx == -1:
                    break

                pivot_doc_id = term_data[pivot_idx]['p'][term_data[pivot_idx]['idx']]
                if pivot_doc_id == -1:
                    break

                if term_data[0]['p'][term_data[0]['idx']] == pivot_doc_id:
                    actual_score = 0.0
                    dl = merged_index.doc_length[pivot_doc_id]
                    for td in term_data:
                        if td['idx'] < td['p_len'] and td['p'][td['idx']] == pivot_doc_id:
                            tf = td['tf'][td['idx']]
                            tf_weight = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * dl / avdl) + tf)
                            actual_score += td['idf'] * tf_weight
                            td['idx'] += 1
                    
                    if len(top_k) < k:
                        heapq.heappush(top_k, (actual_score, self.doc_id_map[pivot_doc_id]))
                    elif actual_score > top_k[0][0]:
                        heapq.heapreplace(top_k, (actual_score, self.doc_id_map[pivot_doc_id]))

                    threshold = top_k[0][0] if len(top_k) == k else 0
                else:
                    it = term_data[0]
                    while it['idx'] < it['p_len'] and it['p'][it['idx']] < pivot_doc_id:
                        it['idx'] += 1

        return sorted(top_k, key=lambda x: (x[0], x[1]), reverse=True)

    def index(self):
        """
        Menjalankan seluruh proses pembuatan indeks dengan skema BSBI.
        Proses ini mencakup parsing blok, pembuatan indeks antara, penyimpanan Kamus,
        dan penggabungan indeks antara menjadi satu indeks utama.
        """
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index()
