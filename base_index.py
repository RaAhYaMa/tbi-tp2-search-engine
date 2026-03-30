import os
import pickle
import contextlib
import heapq
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs, preprocess
from scoring import BM25Scorer, TFIDFScorer

class BaseIndex:
    """
    Kelas dasar untuk pembuatan indeks dan pencarian (retrieval).
    Berisi metode umum untuk pemetaan ID, persistensi metadata, penggabungan (merging),
    dan pencarian peringkat (ranked retrieval).
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        """
        Inisialisasi objek BaseIndex.

        Args:
            data_dir (str): Path ke direktori yang berisi data dokumen.
            output_dir (str): Path ke direktori output untuk menyimpan file indeks.
            postings_encoding: Objek encoding yang digunakan untuk postings list.
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

    def merge(self, indices, merged_index):
        """
        Melakukan penggabungan (merging) beberapa indeks antara menjadi satu indeks tunggal.
        Menggunakan algoritma external merge sort untuk menangani data yang besar.

        Args:
            indices (list): List dari iterator postings list yang akan digabung.
            merged_index (InvertedIndexWriter): Objek writer untuk menyimpan hasil penggabungan.
        """
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        try:
            curr, postings, tf_list = next(merged_iter)
        except StopIteration:
            return

        for t, postings_, tf_list_ in merged_iter:
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)),
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan pencarian dokumen berdasarkan query menggunakan skema pembobotan TF-IDF.

        Args:
            query (str): String query pencarian.
            k (int): Jumlah dokumen teratas yang ingin dikembalikan.

        Returns:
            list: List of tuple (score, doc_name) yang telah diurutkan menurun berdasarkan skor.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        scorer = TFIDFScorer()
        terms = [self.term_id_map[word] for word in preprocess(query) if word in self.term_id_map]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    idf = scorer.idf(N, df)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += scorer.score(tf, idf)

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: (x[0], x[1]), reverse=True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.6, b=0.75):
        """
        Melakukan pencarian dokumen berdasarkan query menggunakan skema pembobotan BM25.

        Args:
            query (str): String query pencarian.
            k (int): Jumlah dokumen teratas yang ingin dikembalikan.
            k1 (float): Parameter k1 untuk BM25 (default: 1.6).
            b (float): Parameter b untuk BM25 (default: 0.75).

        Returns:
            list: List of tuple (score, doc_name) yang telah diurutkan menurun berdasarkan skor.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        scorer = BM25Scorer(k1=k1, b=b)
        terms = [self.term_id_map[word] for word in preprocess(query) if word in self.term_id_map]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            scores = {}
            N = len(merged_index.doc_length)
            avdl = merged_index.avg_doc_length

            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    idf = scorer.idf(N, df)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        dl = merged_index.doc_length[doc_id]
                        scores[doc_id] += scorer.score(tf, idf, dl, avdl)

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: (x[0], x[1]), reverse=True)[:k]

    def retrieve_bm25_wand(self, query, k=10, k1=1.6, b=0.75):
        """
        Melakukan pencarian dokumen menggunakan algoritma WAND (Weak AND) dengan pembobotan BM25.
        Algoritma ini lebih efisien karena dapat melewatkan dokumen yang tidak mungkin 
        masuk ke dalam top-k.

        Args:
            query (str): String query pencarian.
            k (int): Jumlah dokumen teratas yang ingin dikembalikan.
            k1 (float): Parameter k1 untuk BM25.
            b (float): Parameter b untuk BM25.

        Returns:
            list: List of tuple (score, doc_name) yang telah diurutkan menurun berdasarkan skor.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        scorer = BM25Scorer(k1=k1, b=b)
        terms = [self.term_id_map[word] for word in preprocess(query) if word in self.term_id_map]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            avdl = merged_index.avg_doc_length
            
            term_data = []
            min_dl = min(merged_index.doc_length.values()) if merged_index.doc_length else 1
            for term in terms:
                if term in merged_index.postings_dict:
                    data = merged_index.postings_dict[term]
                    max_tf = data[4]
                    df = data[1]
                    idf = scorer.idf(N, df)
                    ut = scorer.upper_bound(max_tf, idf, min_dl, avdl)
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
                if pivot_idx == -1: break
                pivot_doc_id = term_data[pivot_idx]['p'][term_data[pivot_idx]['idx']]
                if pivot_doc_id == -1: break

                if term_data[0]['p'][term_data[0]['idx']] == pivot_doc_id:
                    actual_score = 0.0
                    dl = merged_index.doc_length[pivot_doc_id]
                    for td in term_data:
                        if td['idx'] < td['p_len'] and td['p'][td['idx']] == pivot_doc_id:
                            tf = td['tf'][td['idx']]
                            actual_score += scorer.score(tf, td['idf'], dl, avdl)
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
        Metode abstrak untuk membangun indeks. Harus diimplementasikan oleh subclass.
        """
        raise NotImplementedError("Subclasses must implement index()")
