import re
import math
import argparse
from bsbi import BSBIIndex
from compression import VBEPostings

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


def dcg(ranking):
  score = 0.0
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] / math.log2(i + 1)
  return score

def ndcg(ranking):
  dcg_score = dcg(ranking)
  idcg_score = dcg(sorted(ranking, reverse = True))
  return dcg_score / idcg_score

def ap(ranking):
  score = 0.0
  current_precision = 0.0
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    current_precision = current_precision + (ranking[pos] - current_precision) / i # precision at k
    score += current_precision * ranking[pos]
  return score / len(ranking)

######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

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

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000, metric = 'RBP', scoring = 'tfidf'):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  with open(query_file) as file:
    scores = []
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking = []
      if scoring.lower() == 'tfidf':
        results = BSBI_instance.retrieve_tfidf(query, k = k)
      elif scoring.lower() == 'bm25':
        results = BSBI_instance.retrieve_bm25(query, k = k)
      elif scoring.lower() == 'bm25_wand':
        results = BSBI_instance.retrieve_bm25_wand(query, k = k)
      else:
        raise ValueError("Scoring method tidak dikenal")

      for (score, doc) in results:
          did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
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
  parser.add_argument('--scoring', choices=['tfidf', 'bm25', 'bm25_wand'], default='tfidf', help='Scoring method (tfidf, bm25, bm25_wand)')
  args = parser.parse_args()

  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  eval(qrels, metric=args.metric, scoring=args.scoring)