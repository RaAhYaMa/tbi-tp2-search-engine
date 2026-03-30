import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stemmer = PorterStemmer()
english_stopwords = set(stopwords.words('english'))

def preprocess(text):
    """
    Melakukan normalisasi teks, tokenisasi (hanya karakter alfanumerik), 
    penghapusan stopword, dan stemming (Porter Stemmer).

    Args:
        text (str): Teks mentah yang akan diproses.

    Returns:
        list: Daftar token yang sudah diproses.
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [stemmer.stem(token) for token in tokens if token not in english_stopwords]

class TrieNode:
    """
    Node untuk struktur data Trie.
    Menyimpan anak (children) dalam bentuk dictionary dan ID jika node tersebut adalah akhir kata.
    """
    def __init__(self):
        self.children = {}
        self.id = -1

class Trie:
    """
    Implementasi struktur data Trie untuk pemetaan string ke integer ID secara efisien.
    Sangat berguna untuk menangani kamus term yang besar.
    """
    def __init__(self):
        self.root = TrieNode()

    def __getitem__(self, key):
        """
        Mencari kata dalam Trie dan mengembalikan ID-nya.
        
        Args:
            key (str): Kata yang akan dicari.

        Returns:
            int: ID yang terasosiasi dengan kata tersebut.

        Raises:
            KeyError: Jika kata tidak ditemukan dalam Trie.
        """
        node = self.root
        for char in key:
            if char not in node.children:
                raise KeyError(key)
            node = node.children[char]
        if node.id == -1:
            raise KeyError(key)
        return node.id

    def __setitem__(self, key, value):
        """
        Memasukkan kata baru ke dalam Trie atau memperbarui ID-nya.

        Args:
            key (str): Kata yang akan dimasukkan.
            value (int): ID yang ingin diasosiasikan.
        """
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.id = value

    def __contains__(self, key):
        """Mengecek apakah suatu kata ada di dalam Trie."""
        node = self.root
        for char in key:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.id != -1

class IdMap:
    """
    Kelas untuk mengelola pemetaan dua arah antara string (term atau dokumen) dan integer ID.
    Menggunakan Trie untuk pencarian string-ke-ID yang cepat dan List untuk ID-ke-string.
    """

    def __init__(self):
        """
        Inisialisasi IdMap dengan Trie untuk str_to_id dan List untuk id_to_str.

        Contoh:
            str_to_id["halo"] ---> 8
            id_to_str[8] ---> "halo"
        """
        self.str_to_id = Trie()
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan jumlah total elemen yang dipetakan."""
        return len(self.id_to_str)

    def __get_str(self, i):
        """Mengambil string berdasarkan ID (index)."""
        return self.id_to_str[i]

    def __get_id(self, s):
        """
        Mengambil ID berdasarkan string. Jika string belum ada, ID baru akan di-assign.

        Args:
            s (str): String yang ingin dicari ID-nya.

        Returns:
            int: ID yang unik untuk string tersebut.
        """
        if s not in self.str_to_id:
            self.id_to_str.append(s)
            self.str_to_id[s] = len(self.id_to_str) - 1
        return self.str_to_id[s]

    def __getitem__(self, key):
        """
        Memungkinkan akses menggunakan bracket [..]. 
        Mendukung pencarian dua arah berdasarkan tipe input (int atau str).
        """
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError("Key harus berupa int (untuk ID) atau str (untuk Term/Dokumen)")

def sorted_merge_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Menggabungkan (merge) dua list (doc_id, tf) yang sudah terurut menjadi satu list terurut.
    Akumulasi TF dilakukan jika terdapat doc_id yang sama pada kedua list.

    Contoh: 
        L1 = [(1, 34), (3, 2)]
        L2 = [(1, 11), (2, 4)]
        Hasil = [(1, 45), (2, 4), (3, 2)]

    Args:
        posts_tfs1 (list): List tuple (doc_id, tf) pertama yang sudah terurut.
        posts_tfs2 (list): List tuple (doc_id, tf) kedua yang sudah terurut.

    Returns:
        list: Hasil penggabungan yang sudah terurut berdasarkan doc_id.
    """
    i, j = 0, 0
    merge = []
    while (i < len(posts_tfs1)) and (j < len(posts_tfs2)):
        if posts_tfs1[i][0] == posts_tfs2[j][0]:
            freq = posts_tfs1[i][1] + posts_tfs2[j][1]
            merge.append((posts_tfs1[i][0], freq))
            i += 1
            j += 1
        elif posts_tfs1[i][0] < posts_tfs2[j][0]:
            merge.append(posts_tfs1[i])
            i += 1
        else:
            merge.append(posts_tfs2[j])
            j += 1
    while i < len(posts_tfs1):
        merge.append(posts_tfs1[i])
        i += 1
    while j < len(posts_tfs2):
        merge.append(posts_tfs2[j])
        j += 1
    return merge

def test(output, expected):
    """ simple function for testing """
    return "PASSED" if output == expected else "FAILED"

if __name__ == '__main__':

    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    term_id_map = IdMap()
    assert [term_id_map[term] for term in doc] == [0, 1, 2, 3, 1], "term_id salah"
    assert term_id_map[1] == "semua", "term_id salah"
    assert term_id_map[0] == "halo", "term_id salah"
    assert term_id_map["selamat"] == 2, "term_id salah"
    assert term_id_map["pagi"] == 3, "term_id salah"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname] for docname in docs] == [0, 1, 2], "docs_id salah"

    assert sorted_merge_posts_and_tfs([(1, 34), (3, 2), (4, 23)], \
                                      [(1, 11), (2, 4), (4, 3 ), (6, 13)]) == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], "sorted_merge_posts_and_tfs salah"
    print("All tests passed!")
