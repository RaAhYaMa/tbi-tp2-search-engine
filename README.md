# Information Retrieval System

This project is an implementation of an Information Retrieval (IR) system that includes indexing, searching, and evaluation processes. The system supports various indexing methods, compression techniques, and modern retrieval models.

## Key Features

- **Indexing Methods**:
  - **BSBI (Block-Sort Based Indexing)**: A block-based indexing method designed to handle large collections.
  - **SPIMI (Single-Pass In-Memory Indexing)**: An efficient in-memory indexing method.
- **Postings Compression**:
  - Standard (no compression).
  - **VBE (Variable Byte Encoding)**: Compresses postings and TF lists using variable-length byte encoding.
  - **OptPForDelta**: An advanced compression algorithm for storage efficiency.
- **Search Models (Retrieval Models)**:
  - **TF-IDF**: Classic scoring using term frequency and inverse document frequency.
  - **BM25**: A more advanced probabilistic model for document ranking.
  - **BM25 with WAND (Weak AND)**: Search optimization for fast Top-K query processing.
  - **LSI (Latent Semantic Indexing)**: Semantic-based search using matrix factorization (SVD).
- **Evaluation**:
  - Calculation of evaluation metrics such as **Mean Average Precision (AP/MAP)**, **Rank Biased Precision (RBP)**, **Discounted Cumulative Gain (DCG)**, and **Normalized Discounted Cumulative Gain (NDCG)** based on provided qrels and queries files.

## Project Structure

- `index_construction.py`: Main program to build the index.
- `search_cli.py`: Command Line Interface (CLI) for performing searches.
- `evaluation.py`: Script to evaluate IR system performance or accuracy.
- `bsbi_index.py` & `spimi_index.py`: Logic implementation for BSBI and SPIMI indexing.
- `lsi_index.py`: Implementation of Latent Semantic Indexing.
- `compression.py`: Implementation of compression algorithms (VBE, OptPForDelta).
- `scoring.py`: Implementation of scoring functions (TF-IDF, BM25).
- `util.py`: Utility functions for text processing (tokenization, stemming, etc.).

### Dependency Diagram (ASCII)

```text
[index_construction.py]   [search_cli.py]   [evaluation.py]
          ^                      ^                 ^
          |                      |                 |
+------------------+     +-------------------------------+
|  bsbi_index.py   |     |        base_index.py          |
|  spimi_index.py  | --->|              ^                |
|  lsi_index.py    |     |   (TF-IDF, BM25, WAND)        |
+------------------+     +-------------------------------+
          ^                      ^
          |                      |
+--------------------------------------------------------+
|   index.py (Inverted Index Reader/Writer)              |
+--------------------------------------------------------+
          ^                      ^
          |                      |
+------------------+     +-------------------------------+
|  compression.py  |     |         scoring.py            |
|  (VBE, OptPFor)  |     |      (Ranking Formulas)       |
+------------------+     +-------------------------------+
```

## Installation

Ensure you have Python installed. Install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Build Index

Use `index_construction.py` to build the index from a document collection.

```bash
python index_construction.py --method spimi --compression vbe --lsi
```

Arguments:
- `--method`: `bsbi` or `spimi` (default: `spimi`).
- `--compression`: `standard`, `vbe`, or `optpfordelta` (default: `vbe`).
- `--lsi`: Add this flag to build the LSI index.
- `--data_dir`: Document collection directory (default: `collection`).
- `--output_dir`: Directory for index results (default: `index`).

### 2. Search

Use `search_cli.py` to query against the built index.

```bash
# Single query
python search_cli.py "your query here" --method bm25

# Batch queries from a file
python search_cli.py --file queries.txt --method tfidf
```

Arguments:
- `--method`: `tfidf`, `bm25`, `bm25_wand`, or `lsi` (default: `tfidf`).
- `-k`: Number of top results to return (default: 10).

### 3. Evaluation

Run `evaluation.py` to view the system performance (MAP, RBP, DCG, NDCG).

```bash
python evaluation.py --metric AP --scoring tfidf
```

## System Requirements

- Python 3.x
- Complete library list available in `requirements.txt`.
