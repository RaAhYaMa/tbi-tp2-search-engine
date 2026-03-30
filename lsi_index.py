import os
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import faiss
from base_index import BaseIndex
from index import InvertedIndexReader
from scoring import TFIDFScorer
from tqdm import tqdm
from util import preprocess

class LSIIndex(BaseIndex):
    """
    Class implementing LSI (Latent Semantic Indexing) processing.
    Uses SVD (Singular Value Decomposition) to reduce term-document matrix 
    dimensions and improve semantic search quality.
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index", verbose=True):
        """
        Args:
            data_dir (str): Directory containing document collection.
            output_dir (str): Storage directory for index results.
            postings_encoding (class): Compression class for postings list.
            index_name (str): Base index name.
            verbose (bool): Displays progress logs during index construction.
        """
        super().__init__(data_dir, output_dir, postings_encoding, index_name)
        self.u = None
        self.s = None
        self.vt = None
        self.faiss_index = None
        self.k = None
        self.verbose = verbose

    def build_lsi(self, k=None):
        """
        Builds the LSI (SVD) model and FAISS index for fast vector search.

        Args:
            k (int, optional): Number of reduction dimensions (Singular Values). 
                               If None, k will be automatically calculated based on the collection size.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        num_terms = len(self.term_id_map)
        num_docs = len(self.doc_id_map)

        if k is None:
            k = max(1, min(100, min(num_terms, num_docs) // 2))
        
        if self.verbose:
            print(f"Building LSI matrix: {num_terms} terms, {num_docs} docs")
        
        rows, cols, data = [], [], []
        scorer = TFIDFScorer()
        
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as reader:
            N = len(reader.doc_length)
            for term_id, postings, tf_list in tqdm(reader, desc="Collecting TF-IDF", disable=not self.verbose):
                df = len(postings)
                idf = scorer.idf(N, df)
                for doc_id, tf in zip(postings, tf_list):
                    rows.append(term_id)
                    cols.append(doc_id)
                    weight = scorer.score(tf, idf)
                    data.append(weight)

        A = sp.csr_matrix((data, (rows, cols)), shape=(num_terms, num_docs))
        
        if self.verbose:
            print(f"Performing SVD (k={k})...")
        actual_k = min(k, A.shape[0] - 1, A.shape[1] - 1)
        self.u, self.s, self.vt = svds(A, k=actual_k)
        self.k = actual_k
        
        idx = self.s.argsort()[::-1]
        self.s = self.s[idx]
        self.u = self.u[:, idx]
        self.vt = self.vt[idx, :]
        
        doc_vectors = (self.vt.T @ np.diag(self.s)).astype('float32')
        
        if self.verbose:
            print("Building FAISS index...")
        faiss.normalize_L2(doc_vectors)
        self.faiss_index = faiss.IndexFlatIP(self.k)
        self.faiss_index.add(doc_vectors)
        
        if self.verbose:
            print("LSI Indexing complete.")

    def save_lsi(self):
        """Saves the LSI model and FAISS index to disk."""
        lsi_data = {
            'u': self.u,
            's': self.s,
            'k': self.k
        }
        with open(os.path.join(self.output_dir, self.index_name + '_lsi.model'), 'wb') as f:
            pickle.dump(lsi_data, f)
        
        faiss.write_index(self.faiss_index, os.path.join(self.output_dir, self.index_name + '_lsi.faiss'))
        if self.verbose:
            print("LSI model and FAISS index saved.")

    def load_lsi(self):
        """Loads the LSI model and FAISS index from disk."""
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()
            
        model_path = os.path.join(self.output_dir, self.index_name + '_lsi.model')
        faiss_path = os.path.join(self.output_dir, self.index_name + '_lsi.faiss')
        
        if not os.path.exists(model_path) or not os.path.exists(faiss_path):
            if self.verbose:
                print("LSI model file or FAISS index not found. Building LSI index first...")
            self.build_lsi()
            self.save_lsi()
            return

        with open(model_path, 'rb') as f:
            lsi_data = pickle.load(f)
            self.u = lsi_data['u']
            self.s = lsi_data['s']
            self.k = lsi_data['k']
            
        self.faiss_index = faiss.read_index(faiss_path)

    def retrieve_lsi(self, query, k=10):
        """
        Performs search using the LSI model and FAISS index based on Cosine Similarity values.

        Args:
            query (str): Search query text.
            k (int): Number of top documents to return.

        Returns:
            list: List of (score, docID_path) tuples sorted descending by similarity score.
        """
        if self.faiss_index is None:
            self.load_lsi()
            
        query_words = preprocess(query)
        q_rows, q_cols, q_data = [], [], []
        
        query_tf = {}
        for word in query_words:
            if word in self.term_id_map:
                t_id = self.term_id_map[word]
                query_tf[t_id] = query_tf.get(t_id, 0) + 1
        
        if not query_tf:
            return []
            
        scorer = TFIDFScorer()
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as reader:
            N = len(reader.doc_length)
            for t_id, tf in query_tf.items():
                if t_id in reader.postings_dict:
                    df = reader.postings_dict[t_id][1]
                    idf = scorer.idf(N, df)
                    weight = scorer.score(tf, idf)
                    q_rows.append(t_id)
                    q_cols.append(0)
                    q_data.append(weight)
        
        if not q_data:
            return []
            
        q_sparse = sp.csr_matrix((q_data, (q_rows, q_cols)), shape=(len(self.term_id_map), 1))
        
        q_lsi = (q_sparse.T @ self.u @ np.diag(1.0 / self.s)).astype('float32')
        faiss.normalize_L2(q_lsi)
        scores, indices = self.faiss_index.search(q_lsi, k)
        
        results = []
        for score, doc_id in zip(scores[0], indices[0]):
            if doc_id == -1: continue
            results.append((float(score), self.doc_id_map[int(doc_id)]))
            
        return results

if __name__ == "__main__":
    from compression import VBEPostings
    
    lsi_idx = LSIIndex(data_dir='collection', output_dir='index', postings_encoding=VBEPostings)
    
    lsi_idx.load_lsi()
