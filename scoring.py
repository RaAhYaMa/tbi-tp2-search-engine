import math

class Scorer:
    """
    Kelas dasar untuk semua pemberi skor (scorer).
    Menyediakan fungsionalitas umum seperti perhitungan IDF.
    """
    def idf(self, N, df):
        """
        Menghitung Inverse Document Frequency (IDF) berbasis logaritma natural.
        
        Args:
            N (int): Total dokumen dalam koleksi.
            df (int): Document Frequency (jumlah dokumen yang mengandung term tertentu).

        Returns:
            float: Nilai IDF.
        """
        if df <= 0:
            return 0
        return math.log(N / df)

    def score(self, *args, **kwargs):
        """Metode abstrak untuk menghitung skor akhir."""
        raise NotImplementedError

class TFIDFScorer(Scorer):
    """
    Implementasi skoring TF-IDF.
    Rumus: idf * (1 + log(tf))
    """
    def score(self, tf, idf):
        """
        Menghitung skor TF-IDF untuk satu term dalam sebuah dokumen.

        Args:
            tf (int): Term Frequency dalam dokumen.
            idf (float): Nilai IDF dari term tersebut.

        Returns:
            float: Skor TF-IDF.
        """
        if tf <= 0:
            return 0
        return idf * (1 + math.log(tf))

class BM25Scorer(Scorer):
    """
    Implementasi skoring BM25.
    Rumus: idf * ((k1 + 1) * tf) / (k1 * ((1 - b) + b * dl / avdl) + tf)
    """
    def __init__(self, k1=1.6, b=0.75):
        """
        Args:
            k1 (float): Parameter saturasi TF (default: 1.6).
            b (float): Parameter normalisasi panjang dokumen (default: 0.75).
        """
        self.k1 = k1
        self.b = b

    def tf_weight(self, tf, dl, avdl):
        """
        Menghitung komponen bobot TF pada rumus BM25.

        Args:
            tf (int): Term Frequency.
            dl (int): Panjang dokumen saat ini (Document Length).
            avdl (float): Rata-rata panjang dokumen dalam koleksi (Average Document Length).

        Returns:
            float: Bobot TF hasil normalisasi.
        """
        if tf <= 0:
            return 0
        denominator = self.k1 * ((1 - self.b) + self.b * (dl / avdl)) + tf
        return ((self.k1 + 1) * tf) / denominator

    def score(self, tf, idf, dl, avdl):
        """
        Menghitung skor BM25 lengkap untuk satu term.

        Args:
            tf (int): Term Frequency.
            idf (float): Nilai IDF.
            dl (int): Panjang dokumen.
            avdl (float): Rata-rata panjang dokumen.

        Returns:
            float: Skor BM25.
        """
        return idf * self.tf_weight(tf, dl, avdl)

    def upper_bound(self, max_tf, idf, min_dl, avdl):
        """
        Menghitung batas atas skor (upper bound) untuk optimasi WAND.

        Args:
            max_tf (int): Term Frequency maksimum dari term ini di seluruh koleksi.
            idf (float): Nilai IDF.
            min_dl (int): Panjang dokumen minimum untuk term ini (untuk estimasi konservatif).
            avdl (float): Rata-rata panjang dokumen.

        Returns:
            float: Nilai upper bound skoring.
        """
        return idf * self.tf_weight(max_tf, min_dl, avdl)
