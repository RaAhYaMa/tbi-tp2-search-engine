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
    Performs text normalization, tokenization (alphanumeric characters only), 
    stopword removal, and stemming (Porter Stemmer).

    Args:
        text (str): Raw text to be processed.

    Returns:
        list: List of processed tokens.
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [stemmer.stem(token) for token in tokens if token not in english_stopwords]

class TrieNode:
    """
    Node for the Trie data structure.
    Stores children in a dictionary and an ID if the node is the end of a word.
    """
    def __init__(self):
        self.children = {}
        self.id = -1

class Trie:
    """
    Implementation of the Trie data structure for efficient string-to-integer 
    ID mapping. Very useful for handling large term dictionaries.
    """
    def __init__(self):
        self.root = TrieNode()

    def __getitem__(self, key):
        """
        Searches for a word in the Trie and returns its ID.
        
        Args:
            key (str): The word to search for.

        Returns:
            int: The ID associated with the word.

        Raises:
            KeyError: If the word is not found in the Trie.
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
        Inserts a new word into the Trie or updates its ID.

        Args:
            key (str): The word to be inserted.
            value (int): The ID to be associated.
        """
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.id = value

    def __contains__(self, key):
        """Checks if a word exists in the Trie."""
        node = self.root
        for char in key:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.id != -1

class IdMap:
    """
    Class for managing two-way mapping between strings (terms or documents) 
    and integer IDs. Uses Trie for fast string-to-ID lookup and a list for 
    ID-to-string lookup.
    """

    def __init__(self):
        """
        Initializes IdMap with a Trie for str_to_id and a list for id_to_str.

        Example:
            str_to_id["hello"] ---> 8
            id_to_str[8] ---> "hello"
        """
        self.str_to_id = Trie()
        self.id_to_str = []

    def __len__(self):
        """Returns the total number of mapped elements."""
        return len(self.id_to_str)

    def __get_str(self, i):
        """Retrieves the string based on ID (index)."""
        return self.id_to_str[i]

    def __get_id(self, s):
        """
        Retrieves the ID based on the string. If the string doesn't exist, 
        a new ID will be assigned.

        Args:
            s (str): The string to find the ID for.

        Returns:
            int: The unique ID for the string.
        """
        if s not in self.str_to_id:
            self.id_to_str.append(s)
            self.str_to_id[s] = len(self.id_to_str) - 1
        return self.str_to_id[s]

    def __getitem__(self, key):
        """
        Allows access using brackets [..]. 
        Supports two-way lookup based on input type (int or str).
        """
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError("Key must be an int (for ID) or str (for Term/Document)")

def sorted_merge_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Merges two sorted lists of (doc_id, tf) into a single sorted list.
    TF accumulation is performed if the same doc_id exists in both lists.

    Example: 
        L1 = [(1, 34), (3, 2)]
        L2 = [(1, 11), (2, 4)]
        Result = [(1, 45), (2, 4), (3, 2)]

    Args:
        posts_tfs1 (list): First sorted list of (doc_id, tf) tuples.
        posts_tfs2 (list): Second sorted list of (doc_id, tf) tuples.

    Returns:
        list: The resulting sorted list merged by doc_id.
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
    assert [term_id_map[term] for term in doc] == [0, 1, 2, 3, 1], "incorrect term_id"
    assert term_id_map[1] == "semua", "incorrect term_id"
    assert term_id_map[0] == "halo", "incorrect term_id"
    assert term_id_map["selamat"] == 2, "incorrect term_id"
    assert term_id_map["pagi"] == 3, "incorrect term_id"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname] for docname in docs] == [0, 1, 2], "incorrect docs_id"

    assert sorted_merge_posts_and_tfs([(1, 34), (3, 2), (4, 23)], \
                                      [(1, 11), (2, 4), (4, 3 ), (6, 13)]) == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], "incorrect sorted_merge_posts_and_tfs"
    print("All tests passed!")
