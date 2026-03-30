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
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                # Preprocessing lengkap (case folding, tokenization, stopword removal, stemming)
                for token in preprocess(f.read()):
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
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
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
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
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Preprocessing query agar konsisten dengan index
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

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25(self, query, k = 10, k1 = 1.6, b = 0.75):
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Preprocessing query agar konsisten dengan index
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

            # Top-K
            docs = [(score, self.doc_id_map[doc_id])
                    for (doc_id, score) in scores.items()]

            return sorted(docs, key = lambda x: (x[0], x[1]), reverse = True)[:k]
    
    def retrieve_bm25_wand(self, query, k = 10, k1 = 1.6, b = 0.75):
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Preprocessing query agar konsisten dengan index
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

                    # Untuk dl, loosely pakai dl terpendek dari seluruh dokumen
                    ut = idf * ((k1 + 1) * max_tf) / (k1 * ((1 - b) + b * min_dl / avdl) + max_tf)

                    postings, tf_list = merged_index.get_postings_list(term)
                    p_length = len(postings)
                    postings.append(-1) # Menandakan last_id
                    term_data.append({'p': postings, 'tf': tf_list, 'idx': 0, 'ut': ut, 'idf': idf, 'p_len': p_length})

            if not term_data: return []

            top_k = [] # Min-heap
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
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
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
    BSBI_instance.index() # memulai indexing!
