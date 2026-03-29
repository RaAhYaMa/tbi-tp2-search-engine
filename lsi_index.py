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

class LSIIndex(BaseIndex):
    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index", verbose=True):
        super().__init__(data_dir, output_dir, postings_encoding, index_name)
        self.u = None
        self.s = None
        self.vt = None
        self.faiss_index = None
        self.k = None
        self.verbose = verbose

    def build_lsi(self, k=None):
        """
        Membangun model LSI dan index FAISS dari inverted index yang sudah ada.
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
            # Menggunakan tqdm untuk memantau progres iterasi inverted index
            for term_id, postings, tf_list in tqdm(reader, desc="Collecting TF-IDF", disable=not self.verbose):
                df = len(postings)
                idf = scorer.idf(N, df)
                for doc_id, tf in zip(postings, tf_list):
                    rows.append(term_id)
                    cols.append(doc_id)
                    weight = scorer.score(tf, idf)
                    data.append(weight)

        # Matriks Sparse (Term x Document)
        # Kita gunakan CSR untuk operasi SVD yang efisien
        A = sp.csr_matrix((data, (rows, cols)), shape=(num_terms, num_docs))
        
        if self.verbose:
            print(f"Performing SVD (k={k})...")
        # svds mengembalikan u, s, vt
        # k must be < min(A.shape)
        actual_k = min(k, A.shape[0] - 1, A.shape[1] - 1)
        self.u, self.s, self.vt = svds(A, k=actual_k)
        self.k = actual_k
        
        # Urutkan berdasarkan singular values (svds tidak selalu mengurutkan menurun)
        idx = self.s.argsort()[::-1]
        self.s = self.s[idx]
        self.u = self.u[:, idx]
        self.vt = self.vt[idx, :]
        
        # Vektor dokumen: V * S (setiap kolom V adalah baris Vt)
        # doc_vectors berukuran (num_docs, k)
        doc_vectors = (self.vt.T @ np.diag(self.s)).astype('float32')
        
        if self.verbose:
            print("Building FAISS index...")
        # Normalisasi L2 untuk Cosine Similarity via Inner Product
        faiss.normalize_L2(doc_vectors)
        self.faiss_index = faiss.IndexFlatIP(self.k)
        self.faiss_index.add(doc_vectors)
        
        if self.verbose:
            print("LSI Indexing complete.")

    def save_lsi(self):
        """Menyimpan model LSI dan index FAISS ke disk."""
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
        """Memuat model LSI dan index FAISS dari disk."""
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()
            
        with open(os.path.join(self.output_dir, self.index_name + '_lsi.model'), 'rb') as f:
            lsi_data = pickle.load(f)
            self.u = lsi_data['u']
            self.s = lsi_data['s']
            self.k = lsi_data['k']
            
        self.faiss_index = faiss.read_index(os.path.join(self.output_dir, self.index_name + '_lsi.faiss'))
        # print("LSI model and FAISS index loaded.")

    def retrieve_lsi(self, query, k=10):
        """
        Melakukan pencarian menggunakan model LSI dan FAISS.
        """
        if self.faiss_index is None:
            self.load_lsi()
            
        # 1. Representasikan query sebagai vektor sparse TF-IDF
        query_words = query.split()
        q_rows, q_cols, q_data = [], [], []
        
        # Hitung TF lokal untuk query
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
                    # Query dibentuk sebagai vektor kolom (num_terms x 1)
                    q_rows.append(t_id)
                    q_cols.append(0)
                    q_data.append(weight)
        
        if not q_data:
            return []
            
        q_sparse = sp.csr_matrix((q_data, (q_rows, q_cols)), shape=(len(self.term_id_map), 1))
        
        # 2. Proyeksikan query ke ruang LSI: q_lsi = q^T * U * S^-1
        # q_sparse.T @ self.u menghasilkan matriks (1 x k)
        # Kalikan dengan S^-1 untuk normalisasi skala koordinat
        q_lsi = (q_sparse.T @ self.u @ np.diag(1.0 / self.s)).astype('float32')
        
        # 3. Cari di FAISS
        faiss.normalize_L2(q_lsi)
        scores, indices = self.faiss_index.search(q_lsi, k)
        
        # 4. Format hasil
        results = []
        for score, doc_id in zip(scores[0], indices[0]):
            if doc_id == -1: continue # FAISS returning -1 if not enough results
            results.append((float(score), self.doc_id_map[int(doc_id)]))
            
        return results

if __name__ == "__main__":
    from compression import VBEPostings
    
    # Contoh penggunaan
    lsi_idx = LSIIndex(data_dir='collection', output_dir='index', postings_encoding=VBEPostings)
    
    # Hanya bangun jika belum ada, atau paksa bangun
    if not os.path.exists(os.path.join('index', 'main_index_lsi.model')):
        lsi_idx.build_lsi()
        lsi_idx.save_lsi()
    else:
        lsi_idx.load_lsi()
