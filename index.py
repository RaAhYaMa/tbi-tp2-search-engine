import pickle
import os

class InvertedIndex:
    """
    Kelas yang mengimplementasikan mekanisme pembacaan dan penulisan Inverted Index ke file.
    Dictionary (postings_dict) diasumsikan dapat dimuat seluruhnya di memori.

    Attributes:
        postings_dict (dict): Mapping dari termID ke 5-tuple:
            1. start_position_in_index_file (int): Posisi awal postings list di file index (dalam byte).
            2. number_of_postings_in_list (int): Jumlah docID dalam postings list (Document Frequency).
            3. length_in_bytes_of_postings_list (int): Panjang postings list terkompresi (dalam byte).
            4. length_in_bytes_of_tf_list (int): Panjang TF list terkompresi (dalam byte).
            5. max_tf (int): Nilai term frequency maksimum dalam postings list tersebut.

        terms (list): List of termID untuk menjaga urutan term yang dimasukkan ke index.
    """
    def __init__(self, index_name, postings_encoding, directory=''):
        """
        Args:
            index_name (str): Nama dasar untuk file index dan dictionary.
            postings_encoding (class): Kelas kompresi (misal: VBEPostings).
            directory (str): Lokasi penyimpanan file index.
        """

        self.index_file_path = os.path.join(directory, index_name+'.index')
        self.metadata_file_path = os.path.join(directory, index_name+'.dict')

        self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []
        self.doc_length = {}
        self.avg_doc_length = 0

    def __enter__(self):
        """
        Memuat metadata saat memasuki context manager.
        Metadata yang dimuat meliputi:
            1. postings_dict: Kamus pemetaan termID ke metadata posisi file.
            2. terms: Daftar urutan termID.
            3. doc_length: Kamus panjang dokumen (docID -> jumlah token).
            4. avg_doc_length: Rata-rata panjang dokumen dalam koleksi.
        """
        self.index_file = open(self.index_file_path, 'rb+')

        with open(self.metadata_file_path, 'rb') as f:
            self.postings_dict, self.terms, self.doc_length, self.avg_doc_length = pickle.load(f)
            self.term_iter = self.terms.__iter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Menutup index_file dan menyimpan postings_dict dan terms ketika keluar context"""
        self.index_file.close()

        if self.doc_length:
            self.avg_doc_length = sum(self.doc_length.values()) / len(self.doc_length)
        with open(self.metadata_file_path, 'wb') as f:
            pickle.dump([self.postings_dict, self.terms, self.doc_length, self.avg_doc_length], f)


class InvertedIndexReader(InvertedIndex):
    """
    Class yang mengimplementasikan bagaimana caranya scan atau membaca secara
    efisien Inverted Index yang disimpan di sebuah file.
    """
    def __iter__(self):
        return self

    def reset(self):
        """
        Kembalikan file pointer ke awal, dan kembalikan pointer iterator
        term ke awal
        """
        self.index_file.seek(0)
        self.term_iter = self.terms.__iter__()

    def __next__(self):
        """
        Mengembalikan pasangan (term, postings_list, tf_list) berikutnya secara efisien.
        Hanya memuat bagian kecil dari file index yang dibutuhkan ke memori.
        """
        curr_term = next(self.term_iter)
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf, _ = self.postings_dict[curr_term]
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (curr_term, postings_list, tf_list)

    def get_postings_list(self, term):
        """
        Mengambil postings list dan TF list untuk term tertentu langsung dari posisi byte di file.

        Args:
            term (int/str): termID yang dicari.

        Returns:
            tuple: (postings_list, tf_list).
        """
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf, _ = self.postings_dict[term]
        self.index_file.seek(pos)
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (postings_list, tf_list)


class InvertedIndexWriter(InvertedIndex):
    """
    Class yang mengimplementasikan bagaimana caranya menulis secara
    efisien Inverted Index yang disimpan di sebuah file.
    """
    def __enter__(self):
        self.index_file = open(self.index_file_path, 'wb+')
        return self

    def append(self, term, postings_list, tf_list):
        """
        Menambahkan term, postings list, dan TF list ke akhir file index serta memperbarui metadata.

        Args:
            term (int/str): Identifikasi unik sebuah term.
            postings_list (list): Daftar docID tempat term muncul.
            tf_list (list): Daftar frekuensi term pada dokumen terkait.
        """
        self.terms.append(term)
        for i in range(len(postings_list)):
            doc_id, freq = postings_list[i], tf_list[i]
            if doc_id not in self.doc_length:
                self.doc_length[doc_id] = 0
            self.doc_length[doc_id] += freq

        max_tf = max(tf_list) if tf_list else 0

        self.index_file.seek(0, os.SEEK_END)
        curr_position_in_byte = self.index_file.tell()
        compressed_postings = self.postings_encoding.encode(postings_list)
        compressed_tf_list = self.postings_encoding.encode_tf(tf_list)
        self.index_file.write(compressed_postings)
        self.index_file.write(compressed_tf_list)

        self.postings_dict[term] = (curr_position_in_byte, len(postings_list), \
                                    len(compressed_postings), len(compressed_tf_list), \
                                    max_tf)


if __name__ == "__main__":

    from compression import VBEPostings

    with InvertedIndexWriter('test', postings_encoding=VBEPostings, directory='./tmp/') as index:
        index.append(1, [2, 3, 4, 8, 10], [2, 4, 2, 3, 30])
        index.append(2, [3, 4, 5], [34, 23, 56])
        index.index_file.seek(0)
        assert index.terms == [1,2], "terms salah"
        assert index.doc_length == {2:2, 3:38, 4:25, 5:56, 8:3, 10:30}, "doc_length salah"
        assert index.postings_dict == {1: (0, \
                                           5, \
                                           len(VBEPostings.encode([2,3,4,8,10])), \
                                           len(VBEPostings.encode_tf([2,4,2,3,30])),
                                           30),
                                       2: (len(VBEPostings.encode([2,3,4,8,10])) + len(VBEPostings.encode_tf([2,4,2,3,30])), \
                                           3, \
                                           len(VBEPostings.encode([3,4,5])), \
                                           len(VBEPostings.encode_tf([34,23,56])),
                                           56)}, "postings dictionary salah"
        
        index.index_file.seek(index.postings_dict[2][0])
        assert VBEPostings.decode(index.index_file.read(len(VBEPostings.encode([3,4,5])))) == [3,4,5], "terdapat kesalahan"
        assert VBEPostings.decode_tf(index.index_file.read(len(VBEPostings.encode_tf([34,23,56])))) == [34,23,56], "terdapat kesalahan"
