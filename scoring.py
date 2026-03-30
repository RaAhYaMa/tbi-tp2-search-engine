import math

class Scorer:
    """
    Base class for all scorers.
    Provides common functionality such as IDF calculation.
    """
    def idf(self, N, df):
        """
        Calculates Inverse Document Frequency (IDF) based on the natural logarithm.
        
        Args:
            N (int): Total documents in the collection.
            df (int): Document Frequency (number of documents containing a specific term).

        Returns:
            float: IDF value.
        """
        if df <= 0:
            return 0
        return math.log(N / df)

    def score(self, *args, **kwargs):
        """Abstract method to calculate the final score."""
        raise NotImplementedError

class TFIDFScorer(Scorer):
    """
    TF-IDF scoring implementation.
    Formula: idf * (1 + log(tf))
    """
    def score(self, tf, idf):
        """
        Calculates the TF-IDF score for a single term in a document.

        Args:
            tf (int): Term Frequency in the document.
            idf (float): IDF value for the term.

        Returns:
            float: TF-IDF score.
        """
        if tf <= 0:
            return 0
        return idf * (1 + math.log(tf))

class BM25Scorer(Scorer):
    """
    BM25 scoring implementation.
    Formula: idf * ((k1 + 1) * tf) / (k1 * ((1 - b) + b * dl / avdl) + tf)
    """
    def __init__(self, k1=1.6, b=0.75):
        """
        Args:
            k1 (float): TF saturation parameter (default: 1.6).
            b (float): Document length normalization parameter (default: 0.75).
        """
        self.k1 = k1
        self.b = b

    def tf_weight(self, tf, dl, avdl):
        """
        Calculates the TF weight component in the BM25 formula.

        Args:
            tf (int): Term Frequency.
            dl (int): Current document length (Document Length).
            avdl (float): Average document length in the collection (Average Document Length).

        Returns:
            float: Normalized TF weight.
        """
        if tf <= 0:
            return 0
        denominator = self.k1 * ((1 - self.b) + self.b * (dl / avdl)) + tf
        return ((self.k1 + 1) * tf) / denominator

    def score(self, tf, idf, dl, avdl):
        """
        Calculates the complete BM25 score for a single term.

        Args:
            tf (int): Term Frequency.
            idf (float): IDF value.
            dl (int): Document length.
            avdl (float): Average document length.

        Returns:
            float: BM25 score.
        """
        return idf * self.tf_weight(tf, dl, avdl)

    def upper_bound(self, max_tf, idf, min_dl, avdl):
        """
        Calculates the score upper bound for WAND optimization.

        Args:
            max_tf (int): Maximum Term Frequency of this term across the collection.
            idf (float): IDF value.
            min_dl (int): Minimum document length for this term (for conservative estimation).
            avdl (float): Average document length.

        Returns:
            float: Scoring upper bound value.
        """
        return idf * self.tf_weight(max_tf, min_dl, avdl)
