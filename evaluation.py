import os
import re
import math
import argparse
from base_index import BaseIndex
from lsi_index import LSIIndex
from compression import VBEPostings


def rbp(ranking, p = 0.8):
  """ 
  Menghitung skor Rank Biased Precision (RBP).

  Args:
    ranking (list): List biner [1, 0, ...] yang menunjukkan relevansi dokumen pada tiap peringkat.
    p (float): Parameter persistensi (default: 0.8).

  Returns:
    float: Skor RBP.
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


def dcg(ranking):
  """
  Menghitung skor Discounted Cumulative Gain (DCG).

  Args:
    ranking (list): List biner atau nilai relevansi dokumen pada tiap peringkat.

  Returns:
    float: Skor DCG.
  """
  score = 0.0
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] / math.log2(i + 1)
  return score

def ndcg(ranking):
  """
  Menghitung skor Normalized Discounted Cumulative Gain (NDCG).

  Args:
    ranking (list): List biner atau nilai relevansi dokumen pada tiap peringkat.

  Returns:
    float: Skor NDCG.
  """
  dcg_score = dcg(ranking)
  idcg_score = dcg(sorted(ranking, reverse = True))
  return dcg_score / idcg_score

def ap(ranking):
  """
  Menghitung skor Average Precision (AP).

  Args:
    ranking (list): List biner yang menunjukkan relevansi dokumen pada tiap peringkat.

  Returns:
    float: Skor Average Precision.
  """
  score = 0.0
  current_precision = 0.0
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    current_precision = current_precision + (ranking[pos] - current_precision) / i
    score += current_precision * ranking[pos]
  return score / len(ranking)

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ 
  Memuat data query relevance judgment (qrels) dari file.

  Args:
    qrel_file (str): Path ke file qrels.
    max_q_id (int): Jumlah maksimum query (default: 30).
    max_doc_id (int): Jumlah maksimum dokumen (default: 1033).

  Returns:
    dict: Dictionary bersarang qrels[query_id][doc_id] berisi nilai relevansi (0 atau 1).
  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

def eval(qrels, query_file = "queries.txt", k = 1000, metric = 'RBP', scoring = 'tfidf'):
  """ 
  Melakukan evaluasi performa mesin pencari terhadap seluruh query menggunakan metrik tertentu.
  Menampilkan nilai rata-rata (mean) skor dari semua query.

  Args:
    qrels (dict): Data relevance judgment hasil load_qrels.
    query_file (str): Path ke file yang berisi daftar query.
    k (int): Jumlah dokumen teratas yang diambil untuk evaluasi (default: 1000).
    metric (str): Metrik evaluasi yang digunakan ('RBP', 'DCG', 'NDCG', atau 'AP').
    scoring (str): Metode scoring yang digunakan ('tfidf', 'bm25', 'bm25_wand', atau 'lsi').
  """
  if scoring.lower() == 'lsi':
    index_instance = LSIIndex(data_dir = 'collection', \
                            postings_encoding = VBEPostings, \
                            output_dir = 'index')
    index_instance.load_lsi()
  else:
    index_instance = BaseIndex(data_dir = 'collection', \
                            postings_encoding = VBEPostings, \
                            output_dir = 'index')

  with open(query_file) as file:
    scores = []
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      ranking = []
      if scoring.lower() == 'tfidf':
        results = index_instance.retrieve_tfidf(query, k = k)
      elif scoring.lower() == 'bm25':
        results = index_instance.retrieve_bm25(query, k = k)
      elif scoring.lower() == 'bm25_wand':
        results = index_instance.retrieve_bm25_wand(query, k = k)
      elif scoring.lower() == 'lsi':
        results = index_instance.retrieve_lsi(query, k = k)
      else:
        raise ValueError("Scoring method tidak dikenal")

      for (score, doc) in results:
        match = re.search(r'(\d+)\.txt$', doc)
        if match:
          did = int(match.group(1))
        else:
          did = int(os.path.splitext(os.path.basename(doc))[0])
        ranking.append(qrels[qid][did])
      
      if metric.upper() == 'RBP':
        scores.append(rbp(ranking))
      elif metric.upper() == 'DCG':
        scores.append(dcg(ranking))
      elif metric.upper() == 'NDCG':
        scores.append(ndcg(ranking))
      elif metric.upper() == 'AP':
        scores.append(ap(ranking))
      else:
        raise ValueError("Metric tidak dikenal")

  print(f"Hasil evaluasi {scoring.upper()} dengan metric {metric.upper()} terhadap 30 queries")
  print(f"Mean {metric.upper()} score =", sum(scores) / len(scores))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluasi Search Engine')
  parser.add_argument('--metric', choices=['RBP', 'DCG', 'NDCG', 'AP'], default='RBP', help='Pilihan evaluasi (RBP, DCG, NDCG, AP)')
  parser.add_argument('--scoring', choices=['tfidf', 'bm25', 'bm25_wand', 'lsi'], default='tfidf', help='Scoring method (tfidf, bm25, bm25_wand, lsi)')
  args = parser.parse_args()

  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  eval(qrels, metric=args.metric, scoring=args.scoring)