import os
import re
import math
import argparse
from base_index import BaseIndex
from lsi_index import LSIIndex
from compression import VBEPostings


def rbp(ranking, p = 0.8):
  """ 
  Calculates the Rank Biased Precision (RBP) score.

  Args:
    ranking (list): Binary list [1, 0, ...] indicating document relevance at each rank.
    p (float): Persistence parameter (default: 0.8).

  Returns:
    float: RBP score.
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


def dcg(ranking):
  """
  Calculates the Discounted Cumulative Gain (DCG) score.

  Args:
    ranking (list): Binary list or document relevance values at each rank.

  Returns:
    float: DCG score.
  """
  score = 0.0
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] / math.log2(i + 1)
  return score

def ndcg(ranking):
  """
  Calculates the Normalized Discounted Cumulative Gain (NDCG) score.

  Args:
    ranking (list): Binary list or document relevance values at each rank.

  Returns:
    float: NDCG score.
  """
  dcg_score = dcg(ranking)
  idcg_score = dcg(sorted(ranking, reverse = True))
  return dcg_score / idcg_score

def ap(ranking):
  """
  Calculates the Average Precision (AP) score.

  Args:
    ranking (list): Binary list indicating document relevance at each rank.

  Returns:
    float: Average Precision score.
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
  Loads query relevance judgment (qrels) data from a file.

  Args:
    qrel_file (str): Path to the qrels file.
    max_q_id (int): Maximum number of queries (default: 30).
    max_doc_id (int): Maximum number of documents (default: 1033).

  Returns:
    dict: Nested dictionary qrels[query_id][doc_id] containing relevance values (0 or 1).
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
  Performs search engine performance evaluation against all queries using a specific metric.
  Displays the mean score from all queries.

  Args:
    qrels (dict): Relevance judgment data from load_qrels.
    query_file (str): Path to the file containing the list of queries.
    k (int): Number of top documents retrieved for evaluation (default: 1000).
    metric (str): Evaluation metric used ('RBP', 'DCG', 'NDCG', or 'AP').
    scoring (str): Scoring method used ('tfidf', 'bm25', 'bm25_wand', or 'lsi').
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
        raise ValueError("Unknown scoring method")

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
        raise ValueError("Unknown metric")

  print(f"Evaluation results of {scoring.upper()} with {metric.upper()} metric for 30 queries")
  print(f"Mean {metric.upper()} score =", sum(scores) / len(scores))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Search Engine Evaluation')
  parser.add_argument('--metric', choices=['RBP', 'DCG', 'NDCG', 'AP'], default='RBP', help='Evaluation choices (RBP, DCG, NDCG, AP)')
  parser.add_argument('--scoring', choices=['tfidf', 'bm25', 'bm25_wand', 'lsi'], default='tfidf', help='Scoring method (tfidf, bm25, bm25_wand, lsi)')
  args = parser.parse_args()

  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "incorrect qrels"
  assert qrels["Q1"][300] == 0, "incorrect qrels"

  eval(qrels, metric=args.metric, scoring=args.scoring)