import math

class Scorer:
    """Base class for all scorers."""
    def idf(self, N, df):
        """Standard log-based IDF. Avoid division by zero if df=0."""
        if df == 0:
            return 0
        return math.log(N / df)

    def score(self, *args, **kwargs):
        raise NotImplementedError

class TFIDFScorer(Scorer):
    """
    Implementation of TF-IDF scoring.
    Formula: idf * (1 + log(tf))
    """
    def score(self, tf, idf):
        if tf <= 0:
            return 0
        return idf * (1 + math.log(tf))

class BM25Scorer(Scorer):
    """
    Implementation of BM25 scoring.
    Formula: idf * ((k1 + 1) * tf) / (k1 * ((1 - b) + b * dl / avdl) + tf)
    """
    def __init__(self, k1=1.6, b=0.75):
        self.k1 = k1
        self.b = b

    def tf_weight(self, tf, dl, avdl):
        """Calculates the TF weight component of BM25."""
        if tf <= 0:
            return 0
        denominator = self.k1 * ((1 - self.b) + self.b * (dl / avdl)) + tf
        return ((self.k1 + 1) * tf) / denominator

    def score(self, tf, idf, dl, avdl):
        """Calculates the full BM25 score for a term."""
        return idf * self.tf_weight(tf, dl, avdl)

    def upper_bound(self, max_tf, idf, min_dl, avdl):
        """Calculates the upper bound (ut) for WAND."""
        return idf * self.tf_weight(max_tf, min_dl, avdl)
