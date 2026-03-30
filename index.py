import pickle
import os

class InvertedIndex:
    """
    Class implementing the mechanism for reading and writing the Inverted Index to a file.
    The dictionary (postings_dict) is assumed to fit entirely in memory.

    Attributes:
        postings_dict (dict): Mapping from termID to a 5-tuple:
            1. start_position_in_index_file (int): Start position of the postings list in the index file (in bytes).
            2. number_of_postings_in_list (int): Number of docIDs in the postings list (Document Frequency).
            3. length_in_bytes_of_postings_list (int): Length of the compressed postings list (in bytes).
            4. length_in_bytes_of_tf_list (int): Length of the compressed TF list (in bytes).
            5. max_tf (int): Maximum term frequency value in the postings list.

        terms (list): List of termIDs to maintain the order of terms inserted into the index.
    """
    def __init__(self, index_name, postings_encoding, directory=''):
        """
        Args:
            index_name (str): Base name for the index and dictionary files.
            postings_encoding (class): Compression class (e.g., VBEPostings).
            directory (str): Storage location for the index file.
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
        Loads metadata when entering the context manager.
        The loaded metadata includes:
            1. postings_dict: Dictionary mapping termID to file position metadata.
            2. terms: List of termID sequence.
            3. doc_length: Dictionary of document lengths (docID -> total tokens).
            4. avg_doc_length: Average document length in the collection.
        """
        self.index_file = open(self.index_file_path, 'rb+')

        with open(self.metadata_file_path, 'rb') as f:
            self.postings_dict, self.terms, self.doc_length, self.avg_doc_length = pickle.load(f)
            self.term_iter = self.terms.__iter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Closes index_file and saves postings_dict and terms when exiting the context."""
        self.index_file.close()

        if self.doc_length:
            self.avg_doc_length = sum(self.doc_length.values()) / len(self.doc_length)
        with open(self.metadata_file_path, 'wb') as f:
            pickle.dump([self.postings_dict, self.terms, self.doc_length, self.avg_doc_length], f)


class InvertedIndexReader(InvertedIndex):
    """
    Class implementing efficient scanning or reading of the Inverted Index 
    stored in a file.
    """
    def __iter__(self):
        return self

    def reset(self):
        """
        Resets the file pointer to the beginning and resets the term 
        iterator pointer.
        """
        self.index_file.seek(0)
        self.term_iter = self.terms.__iter__()

    def __next__(self):
        """
        Returns the next (term, postings_list, tf_list) pair efficiently.
        Loads only the required small part of the index file into memory.
        """
        curr_term = next(self.term_iter)
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf, _ = self.postings_dict[curr_term]
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (curr_term, postings_list, tf_list)

    def get_postings_list(self, term):
        """
        Retrieves the postings list and TF list for a specific term directly 
        from the byte position in the file.

        Args:
            term (int/str): The termID to search for.

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
    Class implementing efficient writing of the Inverted Index stored in a file.
    """
    def __enter__(self):
        self.index_file = open(self.index_file_path, 'wb+')
        return self

    def append(self, term, postings_list, tf_list):
        """
        Appends term, postings list, and TF list to the end of the index file 
        and updates metadata.

        Args:
            term (int/str): Unique identification of a term.
            postings_list (list): List of docIDs where the term appears.
            tf_list (list): List of term frequencies in the relevant documents.
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
        assert index.terms == [1,2], "incorrect terms"
        assert index.doc_length == {2:2, 3:38, 4:25, 5:56, 8:3, 10:30}, "incorrect doc_length"
        assert index.postings_dict == {1: (0, \
                                           5, \
                                           len(VBEPostings.encode([2,3,4,8,10])), \
                                           len(VBEPostings.encode_tf([2,4,2,3,30])),
                                           30),
                                       2: (len(VBEPostings.encode([2,3,4,8,10])) + len(VBEPostings.encode_tf([2,4,2,3,30])), \
                                           3, \
                                           len(VBEPostings.encode([3,4,5])), \
                                           len(VBEPostings.encode_tf([34,23,56])),
                                           56)}, "incorrect postings dictionary"
        
        index.index_file.seek(index.postings_dict[2][0])
        assert VBEPostings.decode(index.index_file.read(len(VBEPostings.encode([3,4,5])))) == [3,4,5], "incorrect result"
        assert VBEPostings.decode_tf(index.index_file.read(len(VBEPostings.encode_tf([34,23,56])))) == [34,23,56], "incorrect result"
