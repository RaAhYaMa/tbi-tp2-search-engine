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
    Base class for index creation and retrieval.
    Contains common methods for ID mapping, metadata persistence, merging,
    and ranked retrieval.
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        """
        Initializes the BaseIndex object.

        Args:
            data_dir (str): Path to the directory containing document data.
            output_dir (str): Path to the output directory for storing index files.
            postings_encoding: Encoding object used for the postings list.
            index_name (str): Base name for the generated index files.
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
        Saves term_id_map and doc_id_map to the output directory in pickle format.
        """
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """
        Loads term_id_map and doc_id_map from the output directory.
        """
        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def merge(self, indices, merged_index):
        """
        Performs merging of several intermediate indices into a single index.
        Uses the external merge sort algorithm to handle large data.

        Args:
            indices (list): List of postings list iterators to be merged.
            merged_index (InvertedIndexWriter): Writer object to store the merging results.
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
        Performs document search based on query using the TF-IDF weighting scheme.

        Args:
            query (str): Search query string.
            k (int): Number of top documents to return.

        Returns:
            list: List of (score, doc_name) tuples sorted descending by score.
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
        Performs document search based on query using the BM25 weighting scheme.

        Args:
            query (str): Search query string.
            k (int): Number of top documents to return.
            k1 (float): Parameter k1 for BM25 (default: 1.6).
            b (float): Parameter b for BM25 (default: 0.75).

        Returns:
            list: List of (score, doc_name) tuples sorted descending by score.
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
        Performs document search using the WAND (Weak AND) algorithm with BM25 weighting.
        This algorithm is more efficient as it can skip documents that cannot 
        enter the top-k.

        Args:
            query (str): Search query string.
            k (int): Number of top documents to return.
            k1 (float): Parameter k1 for BM25.
            b (float): Parameter b for BM25.

        Returns:
            list: List of (score, doc_name) tuples sorted descending by score.
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
        Abstract method to build the index. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement index()")
